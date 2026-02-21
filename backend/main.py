"""
Vigilant – RBI Compliance Intelligence API
Main FastAPI application entry point.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional, Union

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
POLICIES_DIR = os.getenv("POLICIES_DIR", "./data/policies")
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default_rbi_config.json"

# Service imports
from services.audio_processor import analyze_audio, check_time_violation
from services.transcriber import transcribe_and_analyze
from services.rag_engine import initialize_policy_store, retrieve_relevant_clauses, get_all_policy_clauses
from services.compliance_engine import run_compliance_analysis
from services.json_builder import build_output_json


# ---------------------------------------------------------------------------
# Default config loader
# ---------------------------------------------------------------------------

def _load_default_config() -> dict:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Supported audio extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm", ".mp4"}


# ---------------------------------------------------------------------------
# FastAPI lifespan – initialize RAG store once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG policy vector store at startup."""
    print("[Vigilant] Starting up – initializing RAG policy store...")
    if not GOOGLE_API_KEY:
        print("[Vigilant] WARNING: GOOGLE_API_KEY not set. RAG and LLM calls will fail.")
    else:
        try:
            initialize_policy_store(POLICIES_DIR, GOOGLE_API_KEY)
            print("[Vigilant] RAG policy store ready.")
        except Exception as exc:
            print(f"[Vigilant] WARNING: Could not initialize RAG store: {exc}")
    yield
    print("[Vigilant] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vigilant – RBI Compliance Intelligence",
    description=(
        "Automated multimodal compliance officer for banks. "
        "Analyzes debt recovery call recordings for policy violations, "
        "emotional tone, and agent conduct. Returns structured JSON audit reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Returns service health status."""
    return {"status": "ok", "service": "Vigilant", "version": "1.0.0"}


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Vigilant – RBI Compliance Intelligence API",
        "docs": "/docs",
        "health": "/health",
        "analyze": "POST /analyze",
    }


@app.get("/config", tags=["Configuration"])
async def get_default_config():
    """
    Returns the default RBI client configuration used when no custom config is provided.
    Use this as a template to build your own client-specific config.
    """
    return _load_default_config()


@app.get("/config/schema", tags=["Configuration"])
async def get_config_schema():
    """
    Returns the JSON schema describing all supported client configuration fields.
    """
    return {
        "description": "Vigilant client configuration schema",
        "fields": {
            "business_domain": {
                "type": "string",
                "description": "The business domain for compliance context (e.g. 'Banking / Debt Recovery', 'Telecom')",
                "example": "Banking / Debt Recovery",
            },
            "monitored_products": {
                "type": "array of strings",
                "description": "Products or services whose mentions should be tracked in calls",
                "example": ["Credit Card", "Personal Loan", "Home Loan"],
            },
            "active_policy_set": {
                "type": "string",
                "description": "Identifier for the policy ruleset to apply",
                "example": "RBI_Compliance_v2.1",
            },
            "risk_triggers": {
                "type": "array of strings",
                "description": "Keywords/phrases that automatically flag a compliance risk when detected",
                "example": ["Legal Threats", "Harassment", "Jail Mention", "Coercion"],
            },
            "custom_rules": {
                "type": "array of objects",
                "description": "Custom policy rules specific to this client, beyond standard RBI clauses",
                "item_schema": {
                    "rule_id": "string — unique identifier e.g. CUSTOM-01",
                    "rule_name": "string — short rule title",
                    "description": "string — full rule description",
                },
                "example": [
                    {
                        "rule_id": "CUSTOM-01",
                        "rule_name": "No Script Deviation",
                        "description": "Agent must follow approved call script at all times.",
                    }
                ],
            },
        },
    }


@app.post("/config/validate", tags=["Configuration"])
async def validate_config(config: dict):
    """
    Validate a client configuration JSON.
    Returns any missing or invalid fields, and the merged effective config.
    """
    default = _load_default_config()
    issues = []

    if "business_domain" in config and not isinstance(config["business_domain"], str):
        issues.append("'business_domain' must be a string.")
    if "monitored_products" in config and not isinstance(config["monitored_products"], list):
        issues.append("'monitored_products' must be an array of strings.")
    if "risk_triggers" in config and not isinstance(config["risk_triggers"], list):
        issues.append("'risk_triggers' must be an array of strings.")
    if "custom_rules" in config:
        if not isinstance(config["custom_rules"], list):
            issues.append("'custom_rules' must be an array of objects.")
        else:
            for i, rule in enumerate(config["custom_rules"]):
                if not isinstance(rule, dict):
                    issues.append(f"'custom_rules[{i}]' must be an object.")
                elif "rule_id" not in rule or "rule_name" not in rule:
                    issues.append(f"'custom_rules[{i}]' must have 'rule_id' and 'rule_name'.")

    merged = {**default, **config}
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "effective_config": merged,
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_call(
    audio_file: Annotated[UploadFile, File(description="Call recording (.mp3/.wav/.ogg/.m4a)")],
    client_config: Annotated[
        Optional[str],
        Form(description="Optional client configuration as a JSON string. If omitted, the default RBI config is used. Use GET /config to see the default template."),
    ] = None,
):
    """
    Analyze a debt recovery call recording for RBI compliance violations.

    - **audio_file**: The call recording (mp3, wav, ogg, m4a, flac supported)
    - **client_config** *(optional)*: JSON string overriding the default client configuration.
      Fields: `business_domain`, `monitored_products`, `active_policy_set`, `risk_triggers`, `custom_rules`.
      Any fields omitted will fall back to the default. See `GET /config` for the full default and `GET /config/schema` for field docs.

    Returns a complete compliance audit JSON report.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY is not configured. Cannot process request.",
        )

    # Validate audio file extension
    suffix = Path(audio_file.filename or "file.mp3").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    # Parse optional client config
    parsed_config: Optional[dict] = None
    if client_config and client_config.strip():
        try:
            parsed_config = json.loads(client_config)
            if not isinstance(parsed_config, dict):
                raise ValueError("client_config must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid client_config JSON: {exc}")

    return await _run_analysis_pipeline(audio_file, parsed_config)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

async def _run_analysis_pipeline(
    audio_upload: UploadFile,
    config_upload: Optional[dict],
) -> JSONResponse:
    """
    Full analysis pipeline:
    1. Save audio to temp file
    2. Load client config (or use default)
    3. Run acoustic analysis + transcription (can overlap)
    4. Check time violation
    5. Retrieve relevant RAG clauses
    6. Run agentic compliance analysis
    7. Assemble and return final JSON
    """
    request_id = f"REQ-{uuid.uuid4().hex[:6].upper()}-MA"
    start_time = time.time()
    tmp_audio_path: Optional[str] = None

    try:
        # -- Step 1: Save uploaded audio to temp file --
        suffix = Path(audio_upload.filename or "audio.mp3").suffix.lower()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=f"vigilant_{request_id}_"
        ) as tmp:
            tmp_audio_path = tmp.name
            content = await audio_upload.read()
            tmp.write(content)
        print(f"[Pipeline] [{request_id}] Audio saved: {tmp_audio_path} ({len(content)} bytes)")

        # -- Step 2: Load config (client overrides merged onto default) --
        default_config = _load_default_config()
        if config_upload is not None:
            # config_upload is now a pre-parsed dict passed from the endpoint
            merged_config = {**default_config, **config_upload}
            # Merge list fields additively (risk_triggers + custom_rules)
            if "risk_triggers" in config_upload:
                merged_config["risk_triggers"] = list({
                    *default_config.get("risk_triggers", []),
                    *config_upload.get("risk_triggers", []),
                })
            if "custom_rules" in config_upload:
                merged_config["custom_rules"] = (
                    default_config.get("custom_rules", []) + config_upload.get("custom_rules", [])
                )
        else:
            merged_config = default_config

        print(f"[Pipeline] [{request_id}] Config: {merged_config.get('active_policy_set')}")

        # -- Step 3a: Acoustic analysis --
        print(f"[Pipeline] [{request_id}] Running acoustic analysis...")
        acoustic_segments = analyze_audio(tmp_audio_path)
        print(f"[Pipeline] [{request_id}] Acoustic: {len(acoustic_segments)} segments")

        # -- Step 3b: Gemini transcription + language detection --
        print(f"[Pipeline] [{request_id}] Running Gemini transcription...")
        transcription_result = transcribe_and_analyze(tmp_audio_path, GOOGLE_API_KEY)
        print(
            f"[Pipeline] [{request_id}] Transcript: "
            f"{len(transcription_result.get('transcript_threads', []))} turns"
        )

        # -- Step 4: Check time violation --
        from datetime import datetime, timezone
        call_timestamp_utc = datetime.now(timezone.utc).isoformat()
        time_violation_result = check_time_violation(call_timestamp_utc)
        if time_violation_result["violation"]:
            print(
                f"[Pipeline] [{request_id}] TIME VIOLATION at {time_violation_result['ist_time']} IST"
            )

        # -- Step 5: Policy clause retrieval (all clauses + RAG for ranking) --
        print(f"[Pipeline] [{request_id}] Loading all policy clauses...")
        all_clauses = get_all_policy_clauses(POLICIES_DIR)
        print(f"[Pipeline] [{request_id}] All clauses loaded: {len(all_clauses)}")

        # Also run RAG retrieval for additional context ranking (merged, deduped)
        rag_clauses = retrieve_relevant_clauses(
            transcript_threads=transcription_result.get("transcript_threads", []),
            api_key=GOOGLE_API_KEY,
            client_config=merged_config,
        )
        # Merge: all_clauses first, then add any RAG clauses not already present
        seen_ids = {c["clause_id"] for c in all_clauses}
        for c in rag_clauses:
            if c.get("clause_id") not in seen_ids:
                all_clauses.append(c)
                seen_ids.add(c["clause_id"])
        print(f"[Pipeline] [{request_id}] Total clauses for compliance check: {len(all_clauses)}")

        # -- Step 6: Agentic compliance analysis --
        print(f"[Pipeline] [{request_id}] Running agentic compliance analysis...")
        compliance_result = run_compliance_analysis(
            transcript_threads=transcription_result.get("transcript_threads", []),
            acoustic_segments=acoustic_segments,
            retrieved_clauses=all_clauses,
            client_config=merged_config,
            call_timestamp_utc=call_timestamp_utc,
            time_violation_result=time_violation_result,
            api_key=GOOGLE_API_KEY,
        )

        # -- Step 7: Assemble final JSON --
        final_output = build_output_json(
            request_id=request_id,
            call_timestamp_utc=call_timestamp_utc,
            processing_start_time=start_time,
            transcription_result=transcription_result,
            acoustic_segments=acoustic_segments,
            compliance_result=compliance_result,
            time_violation_result=time_violation_result,
            client_config=merged_config,
        )

        elapsed = time.time() - start_time
        print(f"[Pipeline] [{request_id}] Complete in {elapsed:.2f}s")
        return JSONResponse(content=final_output)

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[Pipeline] [{request_id}] ERROR: {exc}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(exc)}")
    finally:
        # Clean up temp audio file
        if tmp_audio_path and Path(tmp_audio_path).exists():
            try:
                Path(tmp_audio_path).unlink()
            except Exception:
                pass
