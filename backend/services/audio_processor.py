"""
Audio Preprocessor – Acoustic emotion analysis using librosa.
Extracts energy, pitch, speaking rate, and arousal level per 10-second segment.
Also detects time-based RBI violations (calls outside 8 AM – 7 PM IST).
"""
from __future__ import annotations

import os
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

# Lazy import librosa to avoid slow startup when not needed
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


IST_OFFSET = timedelta(hours=5, minutes=30)
SEGMENT_DURATION_SECONDS = 10
SAMPLE_RATE = 16000

# Arousal thresholds
HIGH_ENERGY_THRESHOLD = 0.65
MEDIUM_ENERGY_THRESHOLD = 0.35
HIGH_PITCH_THRESHOLD = 210.0  # Hz


def _seconds_to_mmss(seconds: float) -> str:
    """Convert floating-point seconds to MM:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _classify_arousal(energy: float, pitch: float) -> str:
    """Map energy + pitch to acoustic arousal label."""
    if energy >= HIGH_ENERGY_THRESHOLD and pitch >= HIGH_PITCH_THRESHOLD:
        return "High"
    elif energy >= MEDIUM_ENERGY_THRESHOLD:
        return "Medium"
    return "Low"


def analyze_audio(file_path: str) -> list[dict]:
    """
    Analyze audio file and return per-segment acoustic features.

    Returns a list of dicts:
    [
        {
            "timestamp": "MM:SS",
            "energy_score": float (0.0–1.0),
            "pitch_hz": float,
            "zcr": float,
            "acoustic_arousal": "Low" | "Medium" | "High"
        },
        ...
    ]
    """
    if not LIBROSA_AVAILABLE:
        # Graceful fallback: return a single generic segment
        return [
            {
                "timestamp": "00:00",
                "energy_score": 0.5,
                "pitch_hz": 150.0,
                "zcr": 0.05,
                "acoustic_arousal": "Medium",
            }
        ]

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        print(f"[AudioProcessor] WARNING: Could not load audio: {exc}")
        return [
            {
                "timestamp": "00:00",
                "energy_score": 0.5,
                "pitch_hz": 150.0,
                "zcr": 0.05,
                "acoustic_arousal": "Medium",
            }
        ]

    segment_samples = SEGMENT_DURATION_SECONDS * sr
    total_samples = len(y)
    num_segments = max(1, math.ceil(total_samples / segment_samples))

    # Global RMS for normalization
    global_rms = float(np.sqrt(np.mean(y ** 2))) or 1e-6

    results: list[dict] = []

    for i in range(num_segments):
        start = i * segment_samples
        end = min(start + segment_samples, total_samples)
        segment = y[start:end]

        if len(segment) < sr * 0.5:
            # Skip very short trailing segments
            continue

        # RMS energy (normalized to 0–1, capped at 1)
        rms = float(np.sqrt(np.mean(segment ** 2)))
        energy_normalized = min(1.0, rms / (global_rms * 2.0 + 1e-6))

        # Fundamental frequency (pitch) via YIN algorithm
        try:
            f0 = librosa.yin(
                segment,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            valid_f0 = f0[f0 > 0]
            pitch = float(np.median(valid_f0)) if len(valid_f0) > 0 else 0.0
        except Exception:
            pitch = 0.0

        # Zero-crossing rate (proxy for speech rate / consonant density)
        zcr_frames = librosa.feature.zero_crossing_rate(segment)
        zcr_mean = float(np.mean(zcr_frames))

        timestamp = _seconds_to_mmss(i * SEGMENT_DURATION_SECONDS)
        arousal = _classify_arousal(energy_normalized, pitch)

        results.append(
            {
                "timestamp": timestamp,
                "energy_score": round(energy_normalized, 4),
                "pitch_hz": round(pitch, 2),
                "zcr": round(zcr_mean, 6),
                "acoustic_arousal": arousal,
            }
        )

    return results if results else [
        {
            "timestamp": "00:00",
            "energy_score": 0.5,
            "pitch_hz": 150.0,
            "zcr": 0.05,
            "acoustic_arousal": "Medium",
        }
    ]


# ---------------------------------------------------------------------------
# Tone / sentiment classification per transcript turn
# ---------------------------------------------------------------------------

# Single-word stems/prefixes: matched using substring-in-text so "harassing" matches "harass" etc.
POSITIVE_STEMS = [
    "thank", "great", "good", "happy", "perfect", "nice", "appreciat",
    "wonderful", "excellent", "okay", "sure", "fine", "alright",
    "help", "assist", "solut", "resolv", "understand", "pleasant",
    "cooperat", "willing", "absolut", "certainly", "of course",
]

ANGRY_STEMS = [
    "angr", "upset", "frustrat", "annoy", "complain",
    "terribl", "horribl", "unacceptabl", "ridicul",
    "useless", "fraud", "scam", "cheat", "liar", "threaten", "threat",
    "demand", "refus", "impossib", "wrong", "mistak",
    "harass", "abuse", "abusiv", "insult", "stupid", "idiot",
    "nonsense", "enough", "fed up", "shut up", "get out",
    "illegal", "police", "jail", "arrest", "legal action",
]

FEARFUL_STEMS = [
    "afraid", "scar", "worr", "anxious", "nervous", "panic",
    "fear", "stress", "concern", "uncertain", "confus", "lost",
    "pleas don", "beg", "mercy", "please don't", "please stop",
]

URGENT_STEMS = [
    "immediately", "asap", "urgent", "right away", "right now",
    "quick", "deadlin", "overd", "final notice", "last chance",
    "warning", "must pay", "pay now", "today itself",
]


def _stem_score(text_lower: str, stems: list[str]) -> int:
    """Count how many stems/phrases appear in the lowercased text."""
    return sum(1 for s in stems if s in text_lower)


def classify_tone(message: str) -> str:
    """
    Classify the emotional tone of a single transcript message.

    Uses substring matching so stemmed forms ("harassing" → "harass") are caught.
    Returns one of: "positive" | "neutral" | "angry/frustrated" | "fearful/anxious" | "urgent"
    """
    text = message.lower()

    angry_score = _stem_score(text, ANGRY_STEMS)
    positive_score = _stem_score(text, POSITIVE_STEMS)
    fearful_score = _stem_score(text, FEARFUL_STEMS)
    urgent_score = _stem_score(text, URGENT_STEMS)

    # Priority: angry > fearful > urgent > positive > neutral
    if angry_score >= 1 and angry_score >= positive_score:
        return "angry/frustrated"
    if fearful_score >= 1:
        return "fearful/anxious"
    if urgent_score >= 1:
        return "urgent"
    if positive_score >= 1:
        return "positive"
    return "neutral"


def add_tone_to_transcript(transcript_threads: list[dict]) -> list[dict]:
    """
    Annotate each turn in transcript_threads with a 'tone' field.

    Mutates in-place and returns the list.
    """
    for turn in transcript_threads:
        turn["tone"] = classify_tone(turn.get("message", ""))
    return transcript_threads


def check_time_violation(call_timestamp_utc: str) -> dict:
    """
    Check whether a call was placed outside RBI-permitted hours (8 AM – 7 PM IST).

    Args:
        call_timestamp_utc: ISO 8601 UTC timestamp string, e.g. "2026-02-20T20:05:12Z"

    Returns:
        {
            "violation": bool,
            "ist_time": "HH:MM",
            "clause_id": "INTERNAL-TIME-01",
            "rule_name": "Operating Hours Compliance",
            "description": str
        }
    """
    try:
        # Parse UTC timestamp
        if call_timestamp_utc.endswith("Z"):
            call_timestamp_utc = call_timestamp_utc[:-1] + "+00:00"
        utc_dt = datetime.fromisoformat(call_timestamp_utc)
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)

        ist_dt = utc_dt.astimezone(timezone(IST_OFFSET))
        hour = ist_dt.hour
        ist_time_str = ist_dt.strftime("%H:%M")

        # RBI rule: 8 AM ≤ call hour < 19 (7 PM)
        is_violation = hour < 8 or hour >= 19

        if is_violation:
            period = "morning (before 8 AM)" if hour < 8 else "evening (after 7 PM)"
            description = (
                f"Call placed outside approved hours (8 AM – 7 PM IST). "
                f"Call was received at {ist_time_str} IST, which is in the {period}."
            )
        else:
            description = f"Call placed within approved hours. IST time: {ist_time_str}."

        return {
            "violation": is_violation,
            "ist_time": ist_time_str,
            "clause_id": "INTERNAL-TIME-01",
            "rule_name": "Operating Hours Compliance",
            "description": description,
        }

    except Exception as exc:
        print(f"[AudioProcessor] WARNING: Could not parse timestamp '{call_timestamp_utc}': {exc}")
        return {
            "violation": False,
            "ist_time": "unknown",
            "clause_id": "INTERNAL-TIME-01",
            "rule_name": "Operating Hours Compliance",
            "description": "Could not determine call time. Timestamp parse error.",
        }
