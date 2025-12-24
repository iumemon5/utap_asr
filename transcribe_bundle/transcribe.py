#!/usr/bin/env python3
"""
Transcribe a single .wav file with Wav2Vec2-CTC (jamo-level) and print Hangul.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)


def create_processor(vocab_path: Path) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        replace_word_delimiter_char="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def load_audio(path: Path, target_sr: int = 16000, min_len: int = 1600) -> np.ndarray:
    speech, sr = sf.read(path)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != target_sr:
        speech = torchaudio.functional.resample(torch.from_numpy(speech).float(), sr, target_sr).numpy()
    if len(speech) < min_len:
        speech = np.pad(speech, (0, min_len - len(speech)), mode="constant")
    return (speech - speech.mean()) / (np.std(speech) + 1e-7)


def normalize_jamo(decoded: str) -> str:
    cleaned = decoded.replace(" ", "")
    words = cleaned.split("|")
    words = [w for w in words if w != ""]
    return " ".join(words)


# Hangul composition helpers
CHO = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JUNG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
DOUBLE_FINAL = {("ㄱ","ㅅ"):"ㄳ", ("ㄴ","ㅈ"):"ㄵ", ("ㄴ","ㅎ"):"ㄶ", ("ㄹ","ㄱ"):"ㄺ", ("ㄹ","ㅁ"):"ㄻ", ("ㄹ","ㅂ"):"ㄼ", ("ㄹ","ㅅ"):"ㄽ", ("ㄹ","ㅌ"):"ㄾ", ("ㄹ","ㅍ"):"ㄿ", ("ㄹ","ㅎ"):"ㅀ", ("ㅂ","ㅅ"):"ㅄ"}


def compose_hangul_from_jamo(jamo_sentence: str) -> str:
    def compose_word(word: str) -> str:
        out = []
        idx = 0
        while idx < len(word):
            ch = word[idx]
            if ch not in CHO:
                out.append(ch)
                idx += 1
                continue
            onset = ch
            if idx + 1 >= len(word) or word[idx + 1] not in JUNG:
                out.append(onset)
                idx += 1
                continue
            vowel = word[idx + 1]
            next_idx = idx + 2
            final = ""
            if next_idx < len(word):
                c1 = word[next_idx]
                if next_idx + 1 < len(word) and (c1, word[next_idx + 1]) in DOUBLE_FINAL and (next_idx + 2 == len(word) or word[next_idx + 2] not in JUNG):
                    final = DOUBLE_FINAL[(c1, word[next_idx + 1])]
                    next_idx += 2
                elif c1 in JONG and (next_idx + 1 == len(word) or word[next_idx + 1] not in JUNG):
                    final = c1
                    next_idx += 1
            code = 0xAC00 + CHO.index(onset) * 588 + JUNG.index(vowel) * 28 + JONG.index(final)
            out.append(chr(code))
            idx = next_idx
        return "".join(out)
    words = [w for w in jamo_sentence.split() if w != ""]
    return " ".join(compose_word(w) for w in words)


def main():
    parser = argparse.ArgumentParser(description="Transcribe a single .wav file with Wav2Vec2-CTC (jamo)")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent / "checkpoint", help="Path to fine-tuned model checkpoint")
    parser.add_argument("--vocab", type=Path, default=Path(__file__).parent / "vocab.json", help="Vocabulary JSON used during training")
    parser.add_argument("--input", type=Path, required=True, help="Path to a single .wav file")
    args = parser.parse_args()

    if not args.input.is_file():
        raise RuntimeError(f"Input must be a .wav file, got: {args.input}")

    processor = create_processor(args.vocab)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir)

    speech = load_audio(args.input)
    # Processor expects raw audio samples; pass as a list for batching
    inputs = processor([speech], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    raw = processor.batch_decode(pred_ids)[0]
    norm = normalize_jamo(raw)
    hangul = compose_hangul_from_jamo(norm)

    print(hangul)


if __name__ == "__main__":
    main()
