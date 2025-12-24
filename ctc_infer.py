#!/usr/bin/env python3
"""Wav2Vec2-CTC inference helper used by the FastAPI service."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2ForCTC

from transcribe_bundle import transcribe as bundle_transcribe


def per_sample_cer(prediction: str, reference: str) -> float:
    """Simple CER implementation to avoid pulling in extra dependencies."""
    if not reference:
        return float(prediction != reference)
    len_ref = len(reference)
    len_pred = len(prediction)
    dp = np.zeros((len_ref + 1, len_pred + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len_ref + 1)
    dp[0, :] = np.arange(len_pred + 1)

    for i in range(1, len_ref + 1):
        for j in range(1, len_pred + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return float(dp[len_ref, len_pred] / max(1, len_ref))


def preprocess_audio(
    audio_bytes: bytes,
    target_sr: int = 16000,
    min_len: int = 1600,
    pad_left_seconds: float = 0.25,
    pad_right_seconds: float = 0.15,
) -> torch.Tensor:
    """Convert raw audio bytes into a normalized mono waveform at the target sample rate."""
    speech, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)  # type: ignore[attr-defined]
    speech = np.asarray(speech, dtype=np.float32)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    if sr != target_sr:
        speech = torchaudio.functional.resample(
            torch.from_numpy(speech).float(),
            sr,
            target_sr,
        ).numpy()

    speech = speech - speech.mean()
    std = float(np.std(speech))
    if std > 0:
        speech = speech / (std + 1e-7)

    pad_left = int(pad_left_seconds * target_sr)
    pad_right = int(pad_right_seconds * target_sr)
    if pad_left or pad_right:
        speech = np.pad(speech, (pad_left, pad_right), mode="constant")

    if len(speech) < min_len:
        speech = np.pad(speech, (0, min_len - len(speech)), mode="constant")

    return torch.tensor(speech, dtype=torch.float32)


@dataclass(frozen=True)
class EngineConfig:
    model_dir: str
    vocab_path: str
    target_sr: int = 16000
    min_len: int = 1600
    language: str = "korean"
    task: str = "transcribe"
    force_cpu: bool = False
    use_half: bool = False


class Wav2Vec2InferenceEngine:
    """Reusable inference helper around the transcribe_bundle Wav2Vec2 model."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.force_cpu else "cpu")
        self.compute_dtype = torch.float16 if (self.device.type == "cuda" and config.use_half) else torch.float32
        self.processor = bundle_transcribe.create_processor(Path(config.vocab_path))
        self.model = Wav2Vec2ForCTC.from_pretrained(config.model_dir)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # allow autotune for convs
        self.model = self.model.to(device=self.device, dtype=self.compute_dtype)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_waveform(
        self,
        waveform: torch.Tensor,
        reference_text: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        speech = waveform.detach().cpu()
        if speech.ndim > 1:
            speech = speech.squeeze(0)
        if speech.ndim > 1:
            speech = speech.mean(dim=0)
        if speech.numel() < self.config.min_len:
            pad = self.config.min_len - speech.numel()
            speech = F.pad(speech, (0, pad))

        inputs = self.processor(
            [speech.numpy()],
            sampling_rate=self.config.target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device, dtype=self.compute_dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits = self.model(input_values=input_values, attention_mask=attention_mask).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw = self.processor.batch_decode(pred_ids)[0]
        normalized = bundle_transcribe.normalize_jamo(raw)
        transcription = bundle_transcribe.compose_hangul_from_jamo(normalized)

        reference_norm = None
        cer_value = None
        if reference_text:
            reference_norm = bundle_transcribe.compose_hangul_from_jamo(
                bundle_transcribe.normalize_jamo(reference_text)
            )
            cer_value = per_sample_cer(transcription, reference_norm)

        return {
            "raw_transcription": raw,
            "transcription_clean": normalized,
            "prediction_norm": transcription,
            "reference_text": reference_text,
            "reference_norm": reference_norm,
            "cer": cer_value,
            "domain_token_enabled": False,
        }


_ENGINE_CACHE: Dict[EngineConfig, Wav2Vec2InferenceEngine] = {}


def get_engine(config: EngineConfig) -> Wav2Vec2InferenceEngine:
    engine = _ENGINE_CACHE.get(config)
    if engine is None:
        engine = Wav2Vec2InferenceEngine(config)
        _ENGINE_CACHE[config] = engine
    return engine
