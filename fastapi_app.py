#!/usr/bin/env python3
"""FastAPI service exposing the Wav2Vec2-CTC inference engine."""

from __future__ import annotations

import asyncio
import base64
from collections import Counter, deque
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ctc_infer import (
    EngineConfig,
    Wav2Vec2InferenceEngine,
    get_engine,
    preprocess_audio,
)

# Configuration
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac", "application/octet-stream"}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_FILE = os.environ.get("LOG_FILE", "api.json")
LOG_BACKUPS = int(os.environ.get("LOG_BACKUPS", "8"))


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "endpoint",
            "audio_seconds",
            "queue_wait_s",
            "preprocess_s",
            "inference_s",
            "postprocess_s",
            "total_s",
            "rtf",
            "status_code",
            "filename",
            "file_count",
            "file_kb",
            "file_size_bytes",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOG_LEVEL)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=str(LOG_DIR / LOG_FILE),
            when="W0",
            interval=1,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
            delay=True,
            utc=False,
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(JsonLogFormatter())
        logger.addHandler(file_handler)
        logger.propagate = False
    return logger


logger = _setup_logging()

EXPECTED_CONDA_ENV = "ctc"
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
QUEUE_WAIT_SECONDS = float(os.environ.get("QUEUE_WAIT_SECONDS", "2.0"))
INFERENCE_TIMEOUT_SECONDS = float(os.environ.get("INFERENCE_TIMEOUT_SECONDS", "30.0"))
MODEL_READY = False
TRANSCRIBE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENCY)
SHUTTING_DOWN = False
INFLIGHT_REQUESTS = 0
INFLIGHT_LOCK = asyncio.Lock()
SHUTDOWN_TIMEOUT_SECONDS = int(os.environ.get("SHUTDOWN_TIMEOUT_SECONDS", "30"))
METRICS_WINDOW = int(os.environ.get("METRICS_WINDOW", "500"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "0") == "1"
EXPECTED_TORCH_VERSION = os.environ.get("EXPECTED_TORCH_VERSION")
EXPECTED_TRANSFORMERS_VERSION = os.environ.get("EXPECTED_TRANSFORMERS_VERSION")
EXPECTED_TORCHAUDIO_VERSION = os.environ.get("EXPECTED_TORCHAUDIO_VERSION")


def ensure_ctc_env() -> None:
    """Warn if the API is not running inside the expected conda env (enforced by run.sh)."""
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env != EXPECTED_CONDA_ENV:
        logger.warning(
            "Expected conda env '%s' but running in '%s'. Activate with: conda activate %s",
            EXPECTED_CONDA_ENV,
            active_env,
            EXPECTED_CONDA_ENV,
        )


ensure_ctc_env()


class Metrics:
    def __init__(self, window: int = 500):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.status_counts = Counter()
        self.latency_ms = deque(maxlen=window)
        self.preprocess_ms = deque(maxlen=window)
        self.inference_ms = deque(maxlen=window)
        self.postprocess_ms = deque(maxlen=window)
        self.queue_wait_ms = deque(maxlen=window)

    def record_success(
        self,
        latency_s: float,
        preprocess_s: float,
        inference_s: float,
        postprocess_s: float,
        queue_wait_s: float,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.latency_ms.append(latency_s * 1000.0)
            self.preprocess_ms.append(preprocess_s * 1000.0)
            self.inference_ms.append(inference_s * 1000.0)
            self.postprocess_ms.append(postprocess_s * 1000.0)
            self.queue_wait_ms.append(queue_wait_s * 1000.0)

    def record_error(self, status_code: int) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_errors += 1
            self.status_counts[status_code] += 1

    def _percentile(self, values: List[float], pct: float) -> Optional[float]:
        if not values:
            return None
        values_sorted = sorted(values)
        idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
        return values_sorted[idx]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latency = list(self.latency_ms)
            preprocess = list(self.preprocess_ms)
            inference = list(self.inference_ms)
            postprocess = list(self.postprocess_ms)
            queue_wait = list(self.queue_wait_ms)
            status_counts = dict(self.status_counts)
            total_requests = self.total_requests
            total_errors = self.total_errors
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "status_counts": status_counts,
            "latency_ms": {
                "p50": self._percentile(latency, 50),
                "p95": self._percentile(latency, 95),
                "p99": self._percentile(latency, 99),
            },
            "preprocess_ms": {
                "p50": self._percentile(preprocess, 50),
                "p95": self._percentile(preprocess, 95),
                "p99": self._percentile(preprocess, 99),
            },
            "inference_ms": {
                "p50": self._percentile(inference, 50),
                "p95": self._percentile(inference, 95),
                "p99": self._percentile(inference, 99),
            },
            "postprocess_ms": {
                "p50": self._percentile(postprocess, 50),
                "p95": self._percentile(postprocess, 95),
                "p99": self._percentile(postprocess, 99),
            },
            "queue_wait_ms": {
                "p50": self._percentile(queue_wait, 50),
                "p95": self._percentile(queue_wait, 95),
                "p99": self._percentile(queue_wait, 99),
            },
            "inflight_requests": INFLIGHT_REQUESTS,
            "max_concurrency": MAX_CONCURRENCY,
        }


METRICS = Metrics(window=METRICS_WINDOW)

APP = FastAPI(
    title="Dysarthria ASR API",
    version="1.0.0",
    description="Speech-to-text API for dysarthric speech using fine-tuned Wav2Vec2 CTC model"
)

# CORS middleware
APP.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@APP.middleware("http")
async def enforce_max_content_length(request, call_next):
    """Early reject requests that advertise too-large bodies to avoid memory pressure."""
    max_bytes = MAX_FILE_SIZE_BYTES
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"Request body too large ({int(content_length) / 1024 / 1024:.1f}MB > "
                    f"{MAX_FILE_SIZE_MB}MB limit)"
                )
            },
        )
    return await call_next(request)


def _build_config() -> EngineConfig:
    model_dir = os.environ.get("CTC_MODEL_DIR", str(Path("transcribe_bundle") / "checkpoint"))
    vocab_path = os.environ.get("CTC_VOCAB_PATH", str(Path("transcribe_bundle") / "vocab.json"))
    target_sr = int(os.environ.get("CTC_TARGET_SR", "16000"))
    min_len = int(os.environ.get("CTC_MIN_SAMPLES", "1600"))
    language = os.environ.get("CTC_LANGUAGE", "korean")
    task = os.environ.get("CTC_TASK", "transcribe")
    force_cpu = os.environ.get("CTC_FORCE_CPU", "0") == "1"
    use_half = os.environ.get("CTC_USE_HALF", "0") == "1"
    return EngineConfig(
        model_dir=model_dir,
        vocab_path=vocab_path,
        target_sr=target_sr,
        min_len=min_len,
        language=language,
        task=task,
        force_cpu=force_cpu,
        use_half=use_half,
    )


def _validate_assets(config: EngineConfig) -> None:
    model_dir = Path(config.model_dir)
    vocab_path = Path(config.vocab_path)
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")
    if not vocab_path.exists():
        raise RuntimeError(f"Vocabulary JSON not found: {vocab_path}")


def _log_startup_config(config: EngineConfig) -> None:
    logger.info(
        "Startup config | model_dir=%s vocab_path=%s target_sr=%s min_len=%s language=%s task=%s force_cpu=%s use_half=%s max_concurrency=%s queue_wait=%ss inference_timeout=%ss shutdown_timeout=%ss",
        config.model_dir,
        config.vocab_path,
        config.target_sr,
        config.min_len,
        config.language,
        config.task,
        config.force_cpu,
        config.use_half,
        MAX_CONCURRENCY,
        QUEUE_WAIT_SECONDS,
        INFERENCE_TIMEOUT_SECONDS,
        SHUTDOWN_TIMEOUT_SECONDS,
    )


def _configure_determinism() -> None:
    if not DETERMINISTIC:
        return
    import numpy as np

    seed = int(os.environ.get("DETERMINISTIC_SEED", "42"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info("Deterministic mode enabled (seed=%s)", seed)


def _check_versions() -> None:
    try:
        import torchaudio
    except Exception:  # pragma: no cover - optional in CPU-only envs
        torchaudio = None  # type: ignore[assignment]
    try:
        import transformers
    except Exception:  # pragma: no cover
        transformers = None  # type: ignore[assignment]

    torch_version = torch.__version__
    transformers_version = getattr(transformers, "__version__", None)
    torchaudio_version = getattr(torchaudio, "__version__", None)
    logger.info(
        "Runtime versions | torch=%s transformers=%s torchaudio=%s",
        torch_version,
        transformers_version,
        torchaudio_version,
    )
    if EXPECTED_TORCH_VERSION and EXPECTED_TORCH_VERSION != torch_version:
        raise RuntimeError(
            f"torch version mismatch: expected {EXPECTED_TORCH_VERSION}, got {torch_version}"
        )
    if EXPECTED_TRANSFORMERS_VERSION and EXPECTED_TRANSFORMERS_VERSION != transformers_version:
        raise RuntimeError(
            "transformers version mismatch: "
            f"expected {EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}"
        )
    if EXPECTED_TORCHAUDIO_VERSION and EXPECTED_TORCHAUDIO_VERSION != torchaudio_version:
        raise RuntimeError(
            f"torchaudio version mismatch: expected {EXPECTED_TORCHAUDIO_VERSION}, got {torchaudio_version}"
        )


APP_CONFIG = _build_config()


def get_engine_dep() -> Wav2Vec2InferenceEngine:
    return get_engine(APP_CONFIG)


class TranscriptionRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio bytes (wav/pcm).")
    reference_text: Optional[str] = Field(None, description="Optional reference CER text.")


class TranscriptionMetadata(BaseModel):
    audio_seconds: float
    language: str
    task: str
    domain_token_enabled: bool
    reference_text: Optional[str]
    cer: Optional[float]
    processing_time_seconds: float
    request_id: str


class TranscriptionResponseDetailed(BaseModel):
    transcription: str = Field(..., description="Normalized transcription output.")
    raw_transcription: Optional[str] = Field(None, description="Raw transcription before normalization.")
    metadata: TranscriptionMetadata


class TranscriptionResponseSimple(BaseModel):
    transcription: str = Field(..., description="Normalized transcription output.")


class BatchTranscriptionItem(BaseModel):
    file_index: int
    filename: str
    transcription: str
    raw_transcription: Optional[str]
    audio_seconds: float
    processing_time_seconds: float
    error: Optional[str]


class BatchTranscriptionResponse(BaseModel):
    request_id: str
    total_files: int
    successful: int
    failed: int
    total_processing_time_seconds: float
    results: List[BatchTranscriptionItem]


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file for size and type."""
    # Check file extension
    if file.filename:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext and file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    # Check content type (optional, as it can be spoofed)
    if file.content_type and file.content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"Suspicious content type: {file.content_type} for file {file.filename}")


@APP.on_event("startup")
async def warmup_engine() -> None:
    """Warm up the model on startup to avoid cold start latency."""
    logger.info("Starting model warmup...")
    engine = get_engine_dep()
    _validate_assets(engine.config)
    _log_startup_config(engine.config)
    _configure_determinism()
    _check_versions()
    try:
        dummy_len = max(engine.config.target_sr, engine.config.min_len)
        dummy = torch.zeros(dummy_len, dtype=torch.float32)
        engine.transcribe_waveform(dummy)
        global MODEL_READY
        MODEL_READY = True
        logger.info("Model warmup completed successfully")
    except Exception as exc:  # pragma: no cover - warmup best effort
        logger.error(f"Warm-up transcription failed: {exc}")


@APP.on_event("shutdown")
async def shutdown_engine() -> None:
    """Stop accepting new work and wait for in-flight requests to finish."""
    global SHUTTING_DOWN
    SHUTTING_DOWN = True
    start = time.time()
    while time.time() - start < SHUTDOWN_TIMEOUT_SECONDS:
        async with INFLIGHT_LOCK:
            if INFLIGHT_REQUESTS == 0:
                logger.info("Shutdown complete: no in-flight requests")
                return
        await asyncio.sleep(0.1)
    logger.warning("Shutdown timeout reached with in-flight requests still running")


@APP.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring and load balancers."""
    if not MODEL_READY:
        logger.warning("Health check requested before model warmup completed")
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        engine = get_engine_dep()
        return {
            "status": "healthy",
            "service": "Dysarthria ASR API",
            "version": "1.0.0"
        }
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@APP.get("/v1/health")
async def health_check_v1() -> Dict[str, Any]:
    """Versioned health check endpoint."""
    return await health_check()


@APP.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check endpoint (process is running)."""
    return {"status": "alive"}


@APP.get("/v1/live")
async def liveness_check_v1() -> Dict[str, Any]:
    """Versioned liveness check endpoint."""
    return await liveness_check()


@APP.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """Lightweight metrics for observability."""
    return METRICS.snapshot()


@APP.get("/v1/metrics")
async def metrics_endpoint_v1() -> Dict[str, Any]:
    """Versioned metrics endpoint."""
    return await metrics_endpoint()


def _process_transcription(
    audio_bytes: bytes,
    engine: Wav2Vec2InferenceEngine,
    reference_text: Optional[str] = None,
    request_id: Optional[str] = None,
    include_raw: bool = False,
    include_metadata: bool = False,
    queue_wait_seconds: float = 0.0,
    endpoint: str = "transcribe",
):
    """Shared transcription logic to avoid code duplication."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    start_time = time.time()
    preprocess_start = time.time()
    try:
        waveform = preprocess_audio(audio_bytes, engine.config.target_sr, engine.config.min_len)
    except Exception as exc:
        logger.error(
            "audio_preprocess_failed request_id=%s error=%s",
            request_id,
            exc,
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
            },
        )
        raise HTTPException(status_code=400, detail="Audio preprocessing failed. Ensure file is valid audio format.") from exc
    preprocess_time = time.time() - preprocess_start

    if waveform.numel() == 0:
        raise HTTPException(status_code=400, detail="Audio data is empty after preprocessing.")

    # Check audio duration (prevent extremely long files)
    audio_seconds = waveform.numel() / float(engine.config.target_sr)
    max_duration = int(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "300"))  # 5 minutes default
    if audio_seconds > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({audio_seconds:.1f}s) exceeds maximum allowed ({max_duration}s)"
        )
    if audio_seconds <= 0:
        raise HTTPException(status_code=400, detail="Audio duration is zero.")
    
    inference_start = time.time()
    try:
        result = engine.transcribe_waveform(
            waveform,
            reference_text=reference_text,
        )
    except Exception as exc:
        logger.error(
            "transcription_failed request_id=%s error=%s",
            request_id,
            exc,
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
            },
        )
        raise HTTPException(status_code=500, detail="Transcription failed. Please try again.") from exc
    inference_time = time.time() - inference_start

    processing_time = time.time() - start_time
    postprocess_time = max(0.0, processing_time - preprocess_time - inference_time)
    rtf = (processing_time / audio_seconds) if audio_seconds > 0 else float("inf")
    
    logger.info(
        "transcribe_complete request_id=%s endpoint=%s audio_seconds=%.2f total_s=%.3f preprocess_s=%.3f inference_s=%.3f postprocess_s=%.3f queue_wait_s=%.3f rtf=%.2f",
        request_id,
        endpoint,
        audio_seconds,
        processing_time,
        preprocess_time,
        inference_time,
        postprocess_time,
        queue_wait_seconds,
        rtf,
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "audio_seconds": audio_seconds,
            "total_s": processing_time,
            "preprocess_s": preprocess_time,
            "inference_s": inference_time,
            "postprocess_s": postprocess_time,
            "queue_wait_s": queue_wait_seconds,
            "rtf": rtf,
        },
    )
    METRICS.record_success(
        latency_s=processing_time,
        preprocess_s=preprocess_time,
        inference_s=inference_time,
        postprocess_s=postprocess_time,
        queue_wait_s=queue_wait_seconds,
    )
    
    # Return simple response by default
    if not include_metadata:
        return TranscriptionResponseSimple(
            transcription=result["prediction_norm"],
        )
    
    # Return detailed response with metadata
    metadata = TranscriptionMetadata(
        audio_seconds=audio_seconds,
        language=engine.config.language,
        task=engine.config.task,
        domain_token_enabled=result["domain_token_enabled"],
        reference_text=result["reference_text"],
        cer=result["cer"],
        processing_time_seconds=round(processing_time, 3),
        request_id=request_id,
    )
    
    return TranscriptionResponseDetailed(
        transcription=result["prediction_norm"],
        raw_transcription=result["raw_transcription"] if include_raw else None,
        metadata=metadata,
    )


def _resolve_request_id(request: Optional[Request]) -> str:
    if request is None:
        return str(uuid.uuid4())
    header_value = request.headers.get("x-request-id")
    return header_value.strip() if header_value else str(uuid.uuid4())


@asynccontextmanager
async def _request_slot(request_id: str):
    if SHUTTING_DOWN:
        METRICS.record_error(status_code=503)
        raise HTTPException(status_code=503, detail="Service is shutting down.")
    if not MODEL_READY:
        METRICS.record_error(status_code=503)
        raise HTTPException(status_code=503, detail="Model not ready.")
    try:
        wait_start = time.time()
        await asyncio.wait_for(TRANSCRIBE_SEMAPHORE.acquire(), timeout=QUEUE_WAIT_SECONDS)
        queue_wait = time.time() - wait_start
    except asyncio.TimeoutError as exc:
        METRICS.record_error(status_code=429)
        raise HTTPException(status_code=429, detail="Server busy. Please retry.") from exc
    global INFLIGHT_REQUESTS
    async with INFLIGHT_LOCK:
        INFLIGHT_REQUESTS += 1
    try:
        yield queue_wait
    finally:
        TRANSCRIBE_SEMAPHORE.release()
        async with INFLIGHT_LOCK:
            INFLIGHT_REQUESTS = max(0, INFLIGHT_REQUESTS - 1)


async def _run_inference(**kwargs):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_process_transcription, **kwargs),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        METRICS.record_error(status_code=504)
        raise HTTPException(status_code=504, detail="Inference timed out.") from exc


@APP.post("/transcribe")
@APP.post("/v1/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    reference_text: Optional[str] = None,
    include_raw: bool = False,
    include_metadata: bool = False,
    engine: Wav2Vec2InferenceEngine = Depends(get_engine_dep),
    request: Request = None,
    response: Response = None,
):
    """Transcribe audio file uploaded via multipart/form-data.
    
    Args:
        file: Audio file (wav, mp3, ogg, flac, m4a)
        reference_text: Optional reference text for CER calculation
        include_raw: Include raw transcription before normalization (requires include_metadata=True)
        include_metadata: Include detailed metadata in response
    
    Returns:
        By default: {"transcription": "..."}
        With include_metadata=True: Full response with metadata
    """
    request_id = _resolve_request_id(request)
    if response is not None:
        response.headers["X-Request-ID"] = request_id
    logger.info(
        "transcribe_start request_id=%s endpoint=transcribe filename=%s",
        request_id,
        file.filename,
        extra={
            "request_id": request_id,
            "endpoint": "transcribe",
            "filename": file.filename,
        },
    )
    
    # Validate file
    validate_audio_file(file)
    
    # Read and validate file size
    try:
        audio_bytes = await file.read()
        file_size = len(audio_bytes)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB}MB)"
            )
        
        logger.info(
            "transcribe_upload request_id=%s endpoint=transcribe file_kb=%.1f",
            request_id,
            file_size / 1024,
            extra={
                "request_id": request_id,
                "endpoint": "transcribe",
                "file_kb": round(file_size / 1024, 1),
                "file_size_bytes": file_size,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "upload_read_failed request_id=%s error=%s",
            request_id,
            exc,
            extra={
                "request_id": request_id,
                "endpoint": "transcribe",
            },
        )
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from exc
    
    async with _request_slot(request_id) as queue_wait:
        try:
            return await _run_inference(
                audio_bytes=audio_bytes,
                engine=engine,
                reference_text=reference_text,
                request_id=request_id,
                include_raw=include_raw,
                include_metadata=include_metadata,
                queue_wait_seconds=queue_wait,
                endpoint="transcribe",
            )
        except HTTPException as exc:
            if exc.status_code not in (429, 504):
                METRICS.record_error(status_code=exc.status_code)
            raise
        except Exception:
            METRICS.record_error(status_code=500)
            raise


@APP.post("/transcribe-base64")
@APP.post("/v1/transcribe-base64")
async def transcribe_base64_endpoint(
    payload: TranscriptionRequest,
    include_raw: bool = False,
    include_metadata: bool = False,
    engine: Wav2Vec2InferenceEngine = Depends(get_engine_dep),
    request: Request = None,
    response: Response = None,
):
    """Transcribe audio from base64-encoded string.
    
    Args:
        payload: TranscriptionRequest with base64 audio and optional reference text
        include_raw: Include raw transcription before normalization (requires include_metadata=True)
        include_metadata: Include detailed metadata in response
    
    Returns:
        By default: {"transcription": "..."}
        With include_metadata=True: Full response with metadata
    """
    request_id = _resolve_request_id(request)
    if response is not None:
        response.headers["X-Request-ID"] = request_id
    logger.info(
        "transcribe_base64_start request_id=%s endpoint=transcribe_base64",
        request_id,
        extra={
            "request_id": request_id,
            "endpoint": "transcribe_base64",
        },
    )
    
    try:
        audio_bytes = base64.b64decode(payload.audio_base64, validate=True)
        file_size = len(audio_bytes)
        
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Decoded audio size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB}MB)"
            )
        logger.info(
            "transcribe_base64_payload request_id=%s endpoint=transcribe_base64 file_kb=%.1f",
            request_id,
            file_size / 1024,
            extra={
                "request_id": request_id,
                "endpoint": "transcribe_base64",
                "file_kb": round(file_size / 1024, 1),
                "file_size_bytes": file_size,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "base64_decode_failed request_id=%s error=%s",
            request_id,
            exc,
            extra={
                "request_id": request_id,
                "endpoint": "transcribe_base64",
            },
        )
        raise HTTPException(status_code=400, detail="Invalid base64 audio data") from exc
    
    async with _request_slot(request_id) as queue_wait:
        try:
            return await _run_inference(
                audio_bytes=audio_bytes,
                engine=engine,
                reference_text=payload.reference_text,
                request_id=request_id,
                include_raw=include_raw,
                include_metadata=include_metadata,
                queue_wait_seconds=queue_wait,
                endpoint="transcribe_base64",
            )
        except HTTPException as exc:
            if exc.status_code not in (429, 504):
                METRICS.record_error(status_code=exc.status_code)
            raise
        except Exception:
            METRICS.record_error(status_code=500)
            raise


@APP.post("/batch-transcribe", response_model=BatchTranscriptionResponse)
@APP.post("/v1/batch-transcribe", response_model=BatchTranscriptionResponse)
async def batch_transcribe_endpoint(
    files: List[UploadFile] = File(..., description="Multiple audio files to transcribe"),
    include_raw: bool = False,
    engine: Wav2Vec2InferenceEngine = Depends(get_engine_dep),
    request: Request = None,
    response: Response = None,
) -> BatchTranscriptionResponse:
    """Transcribe multiple audio files in a batch.
    
    Args:
        files: List of audio files to transcribe
        include_raw: Include raw transcription before normalization
    
    Returns:
        BatchTranscriptionResponse with results for all files
    """
    request_id = _resolve_request_id(request)
    if response is not None:
        response.headers["X-Request-ID"] = request_id
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "10"))
    
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size ({len(files)}) exceeds maximum allowed ({max_batch_size})"
        )
    
    logger.info(
        "batch_transcribe_start request_id=%s endpoint=batch_transcribe file_count=%s",
        request_id,
        len(files),
        extra={
            "request_id": request_id,
            "endpoint": "batch_transcribe",
            "file_count": len(files),
        },
    )
    batch_start_time = time.time()
    
    results: List[BatchTranscriptionItem] = []
    successful = 0
    failed = 0
    
    for idx, file in enumerate(files):
        item_start_time = time.time()
        
        try:
            validate_audio_file(file)
            audio_bytes = await file.read()
            
            if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"File too large: {len(audio_bytes) / 1024 / 1024:.1f}MB")
            
            async with _request_slot(request_id) as queue_wait:
                response = await _run_inference(
                    audio_bytes=audio_bytes,
                    engine=engine,
                    request_id=f"{request_id}-{idx}",
                    include_raw=include_raw,
                    queue_wait_seconds=queue_wait,
                    endpoint="batch_transcribe",
                )
            
            results.append(BatchTranscriptionItem(
                file_index=idx,
                filename=file.filename or f"file_{idx}",
                transcription=response.transcription,
                raw_transcription=response.raw_transcription,
                audio_seconds=response.metadata.audio_seconds,
                processing_time_seconds=round(time.time() - item_start_time, 3),
                error=None,
            ))
            successful += 1
            
        except Exception as exc:
            status_code = 500
            if isinstance(exc, HTTPException):
                status_code = exc.status_code
            METRICS.record_error(status_code=status_code)
            logger.error(
                "batch_item_failed request_id=%s file_index=%s filename=%s error=%s",
                request_id,
                idx,
                file.filename,
                exc,
                extra={
                    "request_id": request_id,
                    "endpoint": "batch_transcribe",
                    "filename": file.filename,
                },
            )
            results.append(BatchTranscriptionItem(
                file_index=idx,
                filename=file.filename or f"file_{idx}",
                transcription="",
                raw_transcription=None,
                audio_seconds=0.0,
                processing_time_seconds=round(time.time() - item_start_time, 3),
                error=str(exc),
            ))
            failed += 1
    
    total_time = time.time() - batch_start_time
    logger.info(f"Batch request {request_id}: Completed in {total_time:.2f}s ({successful} successful, {failed} failed)")
    
    return BatchTranscriptionResponse(
        request_id=request_id,
        total_files=len(files),
        successful=successful,
        failed=failed,
        total_processing_time_seconds=round(total_time, 3),
        results=results,
    )


def dev_main() -> None:
    """Convenience entrypoint for `python fastapi_app.py` during local dev."""
    import uvicorn

    uvicorn.run(
        "fastapi_app:APP",
        host=os.environ.get("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.environ.get("FASTAPI_PORT", "8011")),
        reload=bool(int(os.environ.get("FASTAPI_RELOAD", "0"))),
    )


if __name__ == "__main__":
    dev_main()
