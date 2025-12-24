# UTAP ASR CTC FastAPI Service

This repo contains a FastAPI service that serves a fine-tuned Wav2Vec2 CTC model.

## Setup

1) Create/activate the conda env (default name: `ctc`).
2) Install dependencies (at minimum): `torch`, `torchaudio`, `transformers`, `fastapi`, `uvicorn`, `soundfile`.
3) Place the model checkpoint files in `transcribe_bundle/checkpoint/`.
   - The folder is tracked via `.gitkeep`, but **model files are not committed**.
   - Typical files: `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, etc.

## Run

```bash
chmod +x run.sh
./run.sh
```

Environment variables (optional):
- `CTC_MODEL_DIR` (default: `transcribe_bundle/checkpoint`)
- `CTC_VOCAB_PATH` (default: `transcribe_bundle/vocab.json`)
- `FASTAPI_HOST` / `FASTAPI_PORT`
- `CTC_USE_HALF` (GPU fp16) / `CTC_FORCE_CPU`
- `MAX_CONCURRENCY`, `QUEUE_WAIT_SECONDS`, `INFERENCE_TIMEOUT_SECONDS`
- `LOG_DIR`, `LOG_FILE`, `LOG_LEVEL`

## Test

```bash
curl -X POST "http://localhost:8011/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/sample.wav" \
  -F "include_metadata=true"
```
