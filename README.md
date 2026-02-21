# Vigilant – RBI Compliance Intelligence Platform

> **Automated multimodal compliance officer for Indian banks.**
> Upload a debt recovery call recording → receive a complete AI-generated compliance audit, emotional analysis, policy violation report, and agent performance review — in seconds.

---

## Table of Contents

- [Overview](#overview)
- [Key Features & Benefits](#key-features--benefits)
- [System Architecture](#system-architecture)
- [Complete Workflow](#complete-workflow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Output Schema](#output-schema)
- [Client Configuration](#client-configuration)
- [Policy Documents](#policy-documents)
- [Tech Stack](#tech-stack)

---

## Overview

**Vigilant** is an AI-powered compliance monitoring system purpose-built for Indian banking and NBFC debt recovery operations. It ingests raw call audio and runs a multi-stage analysis pipeline — acoustic signal processing, multilingual transcription, semantic policy retrieval, and LLM reasoning — to produce a structured audit report that a compliance officer or QA team can act on immediately.

| Aspect | Detail |
|---|---|
| **Regulation Target** | RBI Fair Practices Code, NBFC Recovery Guidelines, Internal Call Policies |
| **LLM Backbone** | Google Gemini 1.5 Flash |
| **Vector DB** | ChromaDB (persisted, loaded once at startup) |
| **Throughput** | 1,500 free Gemini requests / day |
| **Latency** | ~10–20 s for a typical 3–5 minute call |
| **Languages** | Malayalam, Hindi, English, Tamil, Telugu, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, and more |

---

## Key Features & Benefits

### 1. Multimodal Audio Intelligence
**What it does:** Processes the raw audio waveform alongside the spoken content rather than just the words.
**Benefit:** Detects emotional escalation, shouting, and high-arousal periods even when the transcript alone looks neutral. Energy, pitch (Hz), and zero-crossing rate are measured every 10 seconds.

### 2. Automatic Multilingual Transcription with Speaker Diarization
**What it does:** Uses Gemini's multimodal API to transcribe the audio, separate agent and customer voices, detect all languages spoken (including mid-call code-switching), and timestamp every utterance.
**Benefit:** Zero manual transcription effort. Handles Tamil–English, Hindi–English, and any mix common in Indian debt recovery calls without a separate language identification step.

### 3. Retrieval-Augmented Generation (RAG) Policy Engine
**What it does:** Every policy clause from RBI guidelines, NBFC fair-practice codes, and internal bank rules is embedded into a ChromaDB vector store. At analysis time, semantically similar clauses are retrieved for each agent utterance.
**Benefit:** The LLM never hallucinates policy requirements. Every violation flag is grounded in an actual, retrievable clause with its source document and clause ID. Policies can be updated simply by editing the `.txt` files — no retraining required.

### 4. Gemini-Powered Compliance Reasoning
**What it does:** A structured prompt feeds the full transcript, acoustic data, all policy clauses, and the client configuration into Gemini 1.5 Flash, which returns the compliance audit as a validated JSON object.
**Benefit:** Replaces manual QA review of calls. Surfaces violations, calculates risk scores, predicts call outcomes, and recommends escalation actions with the same structure every time, ready for downstream dashboards or ticketing systems.

### 5. Time-of-Day Violation Detection
**What it does:** Compares the call timestamp against the RBI-mandated recovery calling window (8 AM – 7 PM IST).
**Benefit:** Automatically flags calls placed outside permitted hours — a common audit finding that is trivially missed in manual review.

### 6. Server-Sent Events (SSE) Streaming
**What it does:** The `POST /analyze/stream` endpoint emits analysis progress in real-time via `text/event-stream`, publishing each stage as it completes.
**Benefit:** Frontend UIs can show a live progress bar (Transcribing → Acoustic Analysis → Policy Retrieval → Compliance Check → Done) rather than waiting on a single long HTTP response.

### 7. Configurable per Bank / Client
**What it does:** Every deployment can override the default policy set with bank-specific `risk_triggers` and `custom_rules` sent as JSON alongside the audio.
**Benefit:** A single Vigilant instance serves multiple banks simultaneously. HDFC, SBI, and a regional NBFC all get different rule sets without separate deployments.

### 8. Automatic Retry with Quota-Aware Backoff
**What it does:** Wraps every Gemini API call in a retry helper that parses the `retry_after` duration from a 429 response and waits exactly that long before retrying, up to 3 attempts.
**Benefit:** Temporary quota bursts do not crash the pipeline. Calls self-heal without any operator intervention.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT  (Browser / API caller)               │
│                                                                      │
│  Upload:  audio_file (.mp3 / .wav / .ogg / .m4a / .flac)           │
│           client_config (optional JSON override)                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ POST /analyze  (or /analyze/stream SSE)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION  (main.py)                    │
│                                                                      │
│   /analyze ──┐                                                       │
│              ├──► _run_analysis_pipeline()   7-step orchestrator     │
│   /analyze   │                                                       │
│   /stream ───┘    /config   /config/schema   /health                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐
  │ AUDIO         │  │ TRANSCRIBER  │  │ RAG ENGINE               │
  │ PROCESSOR     │  │              │  │                          │
  │               │  │ Gemini 1.5   │  │ ChromaDB vector store    │
  │ librosa:      │  │ Flash        │  │ (built once at startup)  │
  │ · Energy      │  │ multimodal   │  │                          │
  │ · Pitch (Hz)  │  │              │  │ Embeddings:              │
  │ · ZCR         │  │ Inline b64   │  │ gemini-embedding-001     │
  │ · Arousal     │  │ (<20MB)      │  │                          │
  │               │  │ Files API    │  │ Sources:                 │
  │ RBI time      │  │ (>20MB)      │  │ · rbi_guidelines.txt     │
  │ window check  │  │              │  │ · nbfc_practices.txt     │
  │ (8AM–7PM IST) │  │ Returns:     │  │ · internal_policies.txt  │
  └───────┬───────┘  │ · Languages  │  │                          │
          │          │ · Transcript │  │ Returns top-k clauses    │
          │          │ · Entities   │  │ semantically matched to  │
          │          │ · Topics     │  │ agent utterances         │
          │          └──────┬───────┘  └────────────┬─────────────┘
          │                 │                        │
          └────────────┬────┘                        │
                       │  all results fed into        │
                       ▼                             │
          ┌────────────────────────────┐             │
          │   COMPLIANCE ENGINE        │◄────────────┘
          │                            │
          │  Gemini 1.5 Flash          │
          │  + _call_with_retry()      │
          │  (429 backoff, 3 retries)  │
          │                            │
          │  Input:                    │
          │  · Transcript threads      │
          │  · Acoustic segments       │
          │  · All policy clauses      │
          │  · Client config           │
          │  · Timestamp + violation   │
          │                            │
          │  Output:                   │
          │  · Compliance flags        │
          │  · Policy violations       │
          │  · Emotional analysis      │
          │  · Risk scores             │
          │  · Agent performance grade │
          │  · Recommended action      │
          └────────────┬───────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │     JSON BUILDER           │
          │                            │
          │  Assembles all outputs     │
          │  into final audit JSON     │
          │  with request_id,          │
          │  metadata, processing_time │
          └────────────┬───────────────┘
                       │
                       ▼
            Final Compliance Audit JSON
```

---

## Complete Workflow

```
STEP 1 — Audio Ingestion
│
│  Client uploads audio file (.mp3 / .wav / .ogg / .m4a / .flac / .webm)
│  Optional: client_config JSON (bank-specific rules)
│  FastAPI saves audio to a temp file, generates unique request_id
▼

STEP 2 — Config Resolution
│
│  Load default_rbi_config.json
│  Merge with client-supplied overrides:
│    · risk_triggers → merged additively (union of default + custom)
│    · custom_rules  → appended to default rules
│    · domain/product fields → replaced by client value
▼

STEP 3a — Acoustic Analysis
│
│  librosa loads the audio at 16 kHz mono
│  Splits into 10-second segments
│  Per segment computes:
│    · RMS energy → normalized energy_score (0.0–1.0)
│    · YIN algorithm → fundamental pitch in Hz
│    · Zero-crossing rate → speech density indicator
│    · (energy + pitch) → arousal level: Low / Medium / High
│  Returns list of {timestamp, energy_score, pitch_hz, zcr, acoustic_arousal}
▼

STEP 3b — Multimodal Transcription (runs alongside Step 3a)
│
│  File size check:
│    ≤ 20 MB → read bytes → base64 encode → send inline with Gemini prompt
│              (fastest path — no upload, no polling)
│    > 20 MB → upload via Gemini Files API → poll until ACTIVE → use URI
│
│  Gemini 1.5 Flash receives audio bytes + transcription prompt
│  Returns JSON containing:
│    · detected_languages   — all languages heard in the call
│    · transcript_threads   — per-turn: speaker (agent/customer), message, MM:SS timestamp
│    · key_topics           — 4–6 main discussion topics
│    · entities             — amounts, dates, account types, persons, locations
│    · primary_intent       — caller/customer intent
│    · root_cause           — reason for the call
│    · sentiment_summary    — brief emotional summary
▼

STEP 4 — Time Violation Check
│
│  Convert UTC call timestamp → IST (UTC+05:30)
│  Check: IST hour ∉ [8, 19)  →  violation = True
│  RBI prohibits debt recovery calls before 8 AM or after 7 PM IST
│  Violation flag and IST time string are passed to compliance engine
▼

STEP 5 — Policy Clause Retrieval (RAG)
│
│  ChromaDB was pre-built at server startup from 3 policy .txt files
│  ~24 individually-chunked clauses, each with:
│    · clause_id (e.g. RBI-RG-03)
│    · rule_name
│    · description
│    · source file name
│
│  All 24 clauses loaded from startup cache (zero disk read per request)
│
│  Semantic search over transcript utterances retrieves top-k most relevant
│  clauses and merges (deduplicates) into the full clause set
▼

STEP 6 — LLM Compliance Reasoning
│
│  Gemini 1.5 Flash receives a structured prompt containing:
│    · Formatted transcript (agent vs customer turns with timestamps)
│    · Acoustic arousal per segment
│    · Full list of all policy clauses (text + clause_id)
│    · Merged client config (risk_triggers, custom_rules, domain, products)
│    · Call timestamp (UTC + IST) and time violation flag
│
│  _call_with_retry() wrapper:
│    · On HTTP 429 → parse "retry in Xs" from error → sleep(X) → retry
│    · Up to 3 attempts before raising
│
│  Gemini returns structured JSON:
│    · compliance_flags      — per-topic True/False flags
│    · policy_violations     — list with clause_id, rule_name, description, severity, source
│    · emotional_analysis    — overall sentiment, tone, tone_progression, emotion_timeline
│    · detected_threats      — legal, coercive, or psychological language found
│    · risk_scores           — compliance_risk, emotional_risk, legal_risk (0.0–1.0)
│    · agent_performance     — professionalism, empathy, script adherence, improvement areas
│    · call_outcome_prediction
│    · recommended_action
▼

STEP 7 — JSON Assembly & Response
│
│  json_builder.assemble() merges all service outputs:
│    · request_id + metadata (languages, processing_time_ms, complexity)
│    · config_applied
│    · intelligence_summary
│    · emotional_and_tonal_analysis
│    · compliance_and_risk_audit
│    · transcript_threads
│    · acoustic_analysis (all segments)
│    · performance_and_outcomes
│
│  Temp audio file deleted from disk
│  JSONResponse returned to caller (or SSE final event for /stream)
▼

RESULT — Structured Compliance Audit Report (JSON)
```

---

## Project Structure

```
vijilant/
├── README.md
│
└── backend/
    ├── main.py                          # FastAPI app, pipeline orchestrator, all endpoints
    ├── requirements.txt                 # Python dependencies
    ├── .env                             # API keys (not committed — add yours here)
    ├── .env.example                     # Placeholder template — safe to commit
    │
    ├── config/
    │   └── default_rbi_config.json      # Default compliance config (risk triggers, products, rules)
    │
    ├── data/
    │   └── policies/
    │       ├── rbi_recovery_guidelines.txt   # RBI Master Directions on recovery
    │       ├── nbfc_fair_practices.txt        # NBFC Fair Practices Code
    │       └── internal_policies.txt          # Internal call center policy clauses
    │
    ├── models/
    │   └── schemas.py                   # Pydantic v2 response models
    │
    ├── services/
    │   ├── audio_processor.py           # librosa acoustic feature extraction + RBI time check
    │   ├── transcriber.py               # Gemini 1.5 Flash multimodal transcription
    │   ├── rag_engine.py                # ChromaDB + LangChain RAG policy pipeline
    │   ├── compliance_engine.py         # Gemini 1.5 Flash compliance reasoning + retry logic
    │   └── json_builder.py              # Final audit JSON assembler
    │
    ├── static/
    │   └── index.html                   # Built-in frontend web UI (served at /)
    │
    └── chroma_db/                       # Auto-generated ChromaDB vector index (gitignored)
        └── rbi_policies/
```

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| FFmpeg | Any recent | Required for MP3/M4A decoding by librosa |
| Google Gemini API Key | — | Free tier at [aistudio.google.com](https://aistudio.google.com/app/apikey) |

**Install FFmpeg**

```bash
# Windows (winget)
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### Installation

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure API Key

Create `backend/.env` (copy from `.env.example`):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Start the Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**First startup** builds the ChromaDB policy vector store (~20–30 seconds).
**Subsequent startups** load the persisted store instantly.

### Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","service":"Vigilant","version":"1.0.0"}
```

- Web UI: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**

---

## API Reference

### `POST /analyze`

Analyze a call recording and receive the full compliance audit report.

| Field | Type | Required | Description |
|---|---|---|---|
| `audio_file` | File (multipart) | ✅ | Call recording: `.mp3` `.wav` `.ogg` `.m4a` `.flac` `.webm` |
| `client_config` | String (multipart) | ❌ | Bank-specific overrides as a JSON string. Omit to use the default RBI config. |

```bash
# Default config
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@call_recording.mp3"

# Custom bank config
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@call_recording.mp3" \
  -F 'client_config={"risk_triggers":["Legal Threats","Jail Mention"],"monitored_products":["Credit Card"]}'
```

---

### `POST /analyze/stream`

Same analysis but streamed as Server-Sent Events. Ideal for live progress UIs.

```bash
curl -N -X POST http://localhost:8000/analyze/stream \
  -F "audio_file=@call_recording.mp3"
```

**Event format:**
```
data: {"stage": "transcription", "status": "running", "message": "Transcribing audio..."}

data: {"stage": "acoustic", "status": "done", "message": "Acoustic analysis complete (18 segments)"}

data: {"stage": "compliance", "status": "done", "result": { ...full audit JSON... }}
```

---

### `GET /health`

```json
{"status": "ok", "service": "Vigilant", "version": "1.0.0"}
```

### `GET /config`

Returns the default RBI client configuration template.

### `GET /config/schema`

Returns documentation for all supported `client_config` fields.

### `POST /config/validate`

Validate a client config JSON before use. Returns validation issues and the merged effective config.

---

## Output Schema

```jsonc
{
  "request_id": "REQ-A3B7F2-MA",
  "metadata": {
    "timestamp": "2025-01-15T09:23:11Z",
    "detected_languages": ["Malayalam", "English"],
    "processing_time_ms": 14230,
    "conversation_complexity": "Medium"
  },
  "config_applied": {
    "business_domain": "Banking / Debt Recovery",
    "monitored_products": ["Credit Card", "Personal Loan"],
    "active_policy_set": "RBI_Compliance_v2.1",
    "risk_triggers": ["Legal Threats", "Harassment", "Jail Mention"]
  },
  "intelligence_summary": {
    "summary": "...",
    "category": "Debt Recovery",
    "conversation_about": "...",
    "primary_intent": "Payment collection",
    "key_topics": ["Overdue EMI", "Settlement Offer"],
    "entities": [{ "text": "₹12,500", "id": "amount_01", "type": "CURRENCY" }],
    "root_cause": "..."
  },
  "emotional_and_tonal_analysis": {
    "overall_sentiment": "Negative",
    "emotional_tone": "Stressed",
    "tone_progression": "Neutral → Tense → Distressed",
    "emotional_graph": [{ "timestamp": "00:30", "emotion": "Anxious", "intensity": 0.7 }],
    "emotion_timeline": "..."
  },
  "compliance_and_risk_audit": {
    "is_within_policy": false,
    "compliance_flags": {
      "time_violation": true,
      "threatening_language": false,
      "harassment": false
    },
    "policy_violations": [
      {
        "clause_id": "RBI-RG-03",
        "rule_name": "Permitted Calling Hours",
        "description": "Call placed at 07:45 IST, before the 8 AM permitted window.",
        "severity": "High",
        "source": "rbi_recovery_guidelines.txt"
      }
    ],
    "detected_threats": [],
    "risk_scores": {
      "compliance_risk": 0.82,
      "emotional_risk": 0.45,
      "legal_risk": 0.60
    }
  },
  "transcript_threads": [
    { "speaker": "agent",    "message": "Hello, this is calling from...", "timestamp": "00:03" },
    { "speaker": "customer", "message": "Yes, who is this?",              "timestamp": "00:07" }
  ],
  "acoustic_analysis": [
    { "timestamp": "00:00", "energy_score": 0.42, "pitch_hz": 198.3, "zcr": 0.08, "acoustic_arousal": "Medium" }
  ],
  "performance_and_outcomes": {
    "agent_performance": {
      "professionalism_score": 0.71,
      "empathy_score": 0.53,
      "script_adherence": "Partial",
      "improvement_areas": ["Avoid early morning calls", "Use de-escalation language"]
    },
    "call_outcome_prediction": "Unresolved – follow-up required",
    "repeat_complaint_detected": false,
    "final_status": "NON-COMPLIANT",
    "recommended_action": "Escalate to QA Manager. Review calling-hours SOP with agent."
  }
}
```

---

## Client Configuration

Pass a JSON string in `client_config` to customise compliance rules per bank:

```json
{
  "business_domain": "Banking / Debt Recovery",
  "monitored_products": ["Credit Card", "Personal Loan", "Home Loan"],
  "active_policy_set": "MY_BANK_v1",
  "risk_triggers": [
    "Legal Threats",
    "Harassment",
    "Jail Mention",
    "Coercion",
    "Family Contact"
  ],
  "custom_rules": [
    {
      "rule_id": "CUSTOM-01",
      "rule_name": "No Script Deviation",
      "description": "Agent must follow the approved call script at all times."
    },
    {
      "rule_id": "CUSTOM-02",
      "rule_name": "No Third-Party Contact",
      "description": "Agent must not discuss the borrower's account with family members or colleagues."
    }
  ]
}
```

Any omitted fields fall back to `config/default_rbi_config.json`.
`risk_triggers` and `custom_rules` are **merged additively**, never replaced.

---

## Policy Documents

Policy documents live in `backend/data/policies/`. Each `.txt` file is chunked into individual clauses at startup and embedded into ChromaDB.

| File | Coverage |
|---|---|
| `rbi_recovery_guidelines.txt` | RBI Master Directions on recovery agents, permitted calling hours, borrower rights |
| `nbfc_fair_practices.txt` | NBFC Ombudsman Scheme, NBFC Fair Practice Code |
| `internal_policies.txt` | Internal call center conduct rules, script adherence, escalation SOP |

To add new policies: drop a `.txt` file into `data/policies/` and restart the server. ChromaDB rebuilds automatically.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API Server | FastAPI + Uvicorn | REST endpoints, SSE streaming, file upload |
| Audio Processing | librosa + NumPy | Acoustic feature extraction (energy, pitch, ZCR, arousal) |
| Transcription | Google Gemini 1.5 Flash (multimodal) | Speech-to-text, speaker diarization, language detection |
| Embeddings | Google `gemini-embedding-001` | Semantic clause embeddings for RAG indexing |
| Vector Store | ChromaDB + LangChain | Policy clause indexing and semantic retrieval |
| Compliance LLM | Google Gemini 1.5 Flash | Policy violation reasoning, emotional analysis, risk scoring |
| Config | python-dotenv | API key and environment variable management |
| Validation | Pydantic v2 | Request and response schema validation |
| Frontend | Vanilla HTML / CSS / JS | Built-in web UI served at `/` |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `POLICIES_DIR` | `./data/policies` | Directory containing policy `.txt` files |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Directory where ChromaDB stores its vector index |

---

*Built for the hackathon track: AI for Financial Compliance & Regulatory Technology.*
