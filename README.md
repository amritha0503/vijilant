# Vigilant – RBI Compliance Intelligence Backend

> Automated multimodal compliance officer for banks.  
> Analyzes debt recovery call recordings for RBI policy violations, emotional tone, and agent conduct.

---

## Quick Start (Local – No Docker)

### 1. Prerequisites
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH (required for MP3 decoding)

### 2. Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Key

Edit `backend/.env`:
```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 4. Run the Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first startup, the RAG policy vector store is built and cached in `./chroma_db/`.  
Subsequent startups load from cache.

### 5. Test the API

Open Swagger UI: http://localhost:8000/docs

**Health check:**
```bash
curl http://localhost:8000/health
```

**Analyze a call recording:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@your_call.mp3"
```

**With custom client config:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@your_call.mp3" \
  -F "client_config=@my_bank_config.json"
```

**Run test suite (requires a test audio file):**
```bash
python test_api.py your_call.mp3
```

---

## Docker

```bash
# From project root
docker-compose up --build
```

API available at: http://localhost:8000

---

## API Reference

### `POST /analyze`

| Field | Type | Required | Description |
|---|---|---|---|
| `audio_file` | File | ✅ | Call recording (.mp3, .wav, .ogg, .m4a, .flac) |
| `client_config` | File | ❌ | Bank-specific JSON policy config |

**Response:** Complete compliance audit JSON (see schema below)

### `GET /health`
Returns `{"status": "ok", "service": "Vigilant"}`

---

## Output JSON Schema

```json
{
  "request_id": "REQ-XXXXXX-MA",
  "metadata": { "timestamp", "detected_languages", "processing_time_ms", "conversation_complexity" },
  "config_applied": { "business_domain", "monitored_products", "active_policy_set", "risk_triggers" },
  "intelligence_summary": { "summary", "category", "conversation_about", "primary_intent", "key_topics", "entities", "root_cause" },
  "emotional_and_tonal_analysis": { "overall_sentiment", "emotional_tone", "tone_progression", "emotional_graph", "emotion_timeline" },
  "compliance_and_risk_audit": { "is_within_policy", "compliance_flags", "policy_violations", "detected_threats", "risk_scores" },
  "transcript_threads": [{ "speaker", "message", "timestamp" }],
  "performance_and_outcomes": { "agent_performance", "call_outcome_prediction", "repeat_complaint_detected", "final_status", "recommended_action" }
}
```

---

## Architecture

```
POST /analyze
     │
     ├── librosa     → Acoustic Analysis (energy, pitch, arousal per segment)
     │
     ├── Gemini 1.5  → Multimodal Transcription (speaker diarization, language detection)
     │
     ├── ChromaDB    → RAG Policy Retrieval (RBI clauses relevant to each agent utterance)
     │
     ├── Gemini 1.5  → Agentic Compliance Reasoning (violations, emotional graph, scores)
     │
     └── JSON Builder → Final structured output
```

---

## Client Config JSON Format

```json
{
  "business_domain": "Banking / Debt Recovery",
  "monitored_products": ["Credit Card", "Personal Loan"],
  "active_policy_set": "MY_BANK_v1",
  "risk_triggers": ["Legal Threats", "Harassment", "Family Mention"],
  "custom_rules": [
    {
      "rule_id": "CUSTOM-01",
      "rule_name": "No Family Contact",
      "description": "Agent must not mention or contact family members."
    }
  ]
}
```

---

## Project Structure

```
backend/
├── main.py                    # FastAPI app + pipeline orchestrator
├── requirements.txt
├── Dockerfile
├── .env                       # API keys (add yours here)
├── test_api.py                # API test suite
├── config/
│   └── default_rbi_config.json
├── data/
│   └── policies/
│       ├── rbi_recovery_guidelines.txt
│       ├── nbfc_fair_practices.txt
│       └── internal_policies.txt
├── models/
│   └── schemas.py             # Pydantic v2 response models
└── services/
    ├── audio_processor.py     # librosa acoustic analysis + time violation check
    ├── transcriber.py         # Gemini multimodal transcription
    ├── rag_engine.py          # ChromaDB + LangChain RAG pipeline
    ├── compliance_engine.py   # Agentic LLM compliance reasoner
    └── json_builder.py        # Final JSON assembler
```
