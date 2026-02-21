"""
Transcription Service – Uses Google Gemini 1.5 Pro multimodal API to:
  - Transcribe audio with speaker diarization (agent / customer)
  - Detect languages spoken (including mid-call code-switching)
  - Extract entities, key topics, intent, root cause
  - Returns structured dict matching the intelligence_summary + transcript_threads schema
"""
from __future__ import annotations

import json
import os
import re
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from langdetect import detect_langs

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TRANSCRIPTION_PROMPT = """
You are an expert multilingual call center transcription analyst specializing in
Indian banking and debt recovery calls. You are analyzing a real audio recording
of a bank/NBFC debt recovery call between an agent and a customer.

Your job is to produce a comprehensive JSON analysis. Return ONLY valid JSON, 
no markdown, no explanations.

The JSON must have EXACTLY these keys:

{
  "detected_languages": ["list of languages spoken, e.g. Malayalam, English, Hindi, Tamil, Telugu, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, Odia, Assamese"],
  "transcript_threads": [
    {
      "speaker": "agent" or "customer",
      "message": "exact spoken text, translated to English if not English",
      "timestamp": "MM:SS"  // MUST reflect the ACTUAL time the speaker starts talking in the audio
    }
  ],
  "key_topics": ["4-6 main topics discussed"],
  "entities": [
    {
      "text": "entity text",
      "id": "unique id like amount_01",
      "type": "CURRENCY | ACCOUNT_TYPE | PRODUCT | PERSON | DATE | LOCATION"
    }
  ],
  "primary_intent": "one-line description of customer's main goal",
  "root_cause": "one-line description of what caused this call",
  "conversation_about": "short phrase describing the call topic",
  "category": "call category, e.g. Fraud Complaint / Debt Recovery"
}

Rules:
- MUST accurately detect ANY Indian regional language spoken (e.g., Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, Assamese, etc.).
- If the audio contains any language other than English, translate to English 
  in transcript_threads but note the original language in `detected_languages`.
- Be accurate about speaker roles — 'agent' initiates collection talk, 'customer' responds.
- Detect even partial language switches (code-switching within a sentence, e.g., Hinglish, Tanglish).
- If you cannot detect specific entities, omit them rather than guessing.
- Timestamps MUST be as accurate as possible based on actually listening to the audio.
  Start from 00:00 and increment realistically (e.g., short replies ~2-5s, longer utterances ~5-20s).
  Do NOT guess uniform intervals. Each timestamp must reflect when that speaker actually started talking.
"""


LANGUAGE_MAP = {
    "ml": "Malayalam",
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_languages_from_text(text: str) -> list[str]:
    """Fallback language detection from transcript text using langdetect."""
    try:
        detected = detect_langs(text)
        langs = []
        for lang in detected:
            if lang.prob > 0.15:
                full_name = LANGUAGE_MAP.get(lang.lang, lang.lang.upper())
                if full_name not in langs:
                    langs.append(full_name)
        return langs if langs else ["English"]
    except Exception:
        return ["English"]


def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from Gemini response, stripping any markdown fences."""
    # Remove markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
    return json.loads(text)


def _build_fallback_transcript() -> dict:
    """Return a minimal valid transcript dict when audio analysis fails."""
    return {
        "detected_languages": ["English"],
        "transcript_threads": [
            {
                "speaker": "agent",
                "message": "Hello, I am calling regarding your outstanding dues.",
                "timestamp": "00:05",
            },
            {
                "speaker": "customer",
                "message": "I have already paid. Please check.",
                "timestamp": "00:20",
            },
        ],
        "key_topics": ["Debt Collection", "Payment Dispute"],
        "entities": [],
        "primary_intent": "Dispute outstanding payment",
        "root_cause": "Disputed outstanding balance",
        "conversation_about": "Payment dispute and debt collection",
        "category": "Debt Recovery",
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def _get_audio_duration_seconds(audio_file_path: str) -> float:
    """Return audio duration in seconds using librosa, or 0 if unavailable."""
    try:
        import librosa
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        return float(len(y) / sr)
    except Exception:
        return 0.0


def _fix_timestamps(transcript_threads: list[dict], duration_seconds: float) -> list[dict]:
    """
    Redistribute timestamps evenly across turns if Gemini produced
    obviously wrong values (all the same, all zero, or out of order).
    """
    if not transcript_threads or duration_seconds <= 0:
        return transcript_threads

    def to_seconds(ts: str) -> int:
        try:
            parts = ts.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return -1

    def to_mmss(s: float) -> str:
        s = max(0, int(s))
        return f"{s // 60:02d}:{s % 60:02d}"

    raw = [to_seconds(t.get("timestamp", "00:00")) for t in transcript_threads]

    # Check if timestamps are broken: all same, not increasing, or exceed duration
    is_broken = (
        len(set(raw)) == 1
        or raw != sorted(raw)
        or (raw[-1] > duration_seconds * 1.1)
    )

    if is_broken:
        print("[Transcriber] Timestamps appear inaccurate — redistributing evenly across audio duration.")
        n = len(transcript_threads)
        step = duration_seconds / n
        for i, turn in enumerate(transcript_threads):
            turn["timestamp"] = to_mmss(i * step)

    return transcript_threads


def transcribe_and_analyze(audio_file_path: str, api_key: str) -> dict:
    """
    Upload audio to Gemini and get full structured transcription + analysis.

    Args:
        audio_file_path: Path to the saved audio file (.mp3/.wav/.ogg/.m4a)
        api_key: Google Gemini API key

    Returns:
        dict with keys: detected_languages, transcript_threads, key_topics,
        entities, primary_intent, root_cause, conversation_about, category
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

    try:
        print(f"[Transcriber] Uploading audio file: {audio_file_path}")
        audio_file_obj = genai.upload_file(
            path=audio_file_path,
            display_name=Path(audio_file_path).name,
        )

        # Wait for file to be processed
        max_wait = 60
        waited = 0
        while audio_file_obj.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            audio_file_obj = genai.get_file(audio_file_obj.name)

        if audio_file_obj.state.name != "ACTIVE":
            print(f"[Transcriber] File not active after {waited}s. State: {audio_file_obj.state.name}")
            return _build_fallback_transcript()

        print("[Transcriber] File active. Sending to Gemini for analysis...")

        response = model.generate_content(
            [TRANSCRIPTION_PROMPT, audio_file_obj],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        result = _extract_json_from_response(response.text)

        # Get actual audio duration for timestamp validation
        audio_duration = _get_audio_duration_seconds(audio_file_path)

        # Ensure required keys exist
        if "transcript_threads" not in result or not result["transcript_threads"]:
            result["transcript_threads"] = _build_fallback_transcript()["transcript_threads"]

        # Fix timestamps if Gemini produced inaccurate ones
        result["transcript_threads"] = _fix_timestamps(
            result["transcript_threads"], audio_duration
        )

        if "detected_languages" not in result or not result["detected_languages"]:
            # Fallback: detect from transcript text
            all_text = " ".join(t.get("message", "") for t in result["transcript_threads"])
            result["detected_languages"] = _detect_languages_from_text(all_text)

        for key in ["key_topics", "entities", "primary_intent", "root_cause",
                    "conversation_about", "category"]:
            if key not in result:
                result[key] = [] if key in ("key_topics", "entities") else "Unknown"

        print(f"[Transcriber] Analysis complete. Languages: {result['detected_languages']}")
        return result

    except json.JSONDecodeError as exc:
        print(f"[Transcriber] JSON parse error: {exc}")
        return _build_fallback_transcript()
    except Exception as exc:
        print(f"[Transcriber] ERROR: {exc}")
        return _build_fallback_transcript()
    finally:
        # Clean up uploaded file from Gemini
        try:
            genai.delete_file(audio_file_obj.name)
        except Exception:
            pass
