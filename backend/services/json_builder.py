"""
JSON Output Builder – Assembles the final Vigilant response JSON
from all service outputs, matching the exact output schema.
"""
from __future__ import annotations

import time
from typing import Optional

from services.audio_processor import add_tone_to_transcript


def _complexity_from_turns(turn_count: int) -> str:
    if turn_count <= 6:
        return "low"
    elif turn_count <= 14:
        return "medium"
    return "high"


def build_output_json(
    request_id: str,
    call_timestamp_utc: str,
    processing_start_time: float,
    transcription_result: dict,
    acoustic_segments: list[dict],
    compliance_result: dict,
    time_violation_result: dict,
    client_config: dict,
) -> dict:
    """
    Assemble the complete Vigilant response JSON.

    Args:
        request_id: Unique request identifier
        call_timestamp_utc: Original call timestamp (from metadata or current time)
        processing_start_time: time.time() at start of pipeline
        transcription_result: Output from transcriber.py
        acoustic_segments: Output from audio_processor.py
        compliance_result: Output from compliance_engine.py
        time_violation_result: Output from audio_processor.check_time_violation()
        client_config: The merged client+default configuration dict

    Returns:
        Complete dict matching the Vigilant output JSON schema
    """
    processing_ms = int((time.time() - processing_start_time) * 1000)
    transcript_threads = add_tone_to_transcript(
        transcription_result.get("transcript_threads", [])
    )

    # ---- metadata ----
    metadata = {
        "timestamp": call_timestamp_utc,
        "detected_languages": transcription_result.get("detected_languages", ["English"]),
        "processing_time_ms": processing_ms,
        "conversation_complexity": _complexity_from_turns(len(transcript_threads)),
    }

    # ---- config_applied ----
    config_applied = {
        "business_domain": client_config.get("business_domain", "Banking / Debt Recovery"),
        "monitored_products": client_config.get("monitored_products", []),
        "active_policy_set": client_config.get("active_policy_set", "RBI_Compliance_v2.1"),
        "risk_triggers": client_config.get("risk_triggers", []),
    }

    # ---- intelligence_summary ----
    entities = transcription_result.get("entities", [])
    # Ensure each entity has required fields
    cleaned_entities = []
    for i, entity in enumerate(entities):
        cleaned_entities.append({
            "text": entity.get("text", ""),
            "id": entity.get("id", f"entity_{i:02d}"),
            "type": entity.get("type", "UNKNOWN"),
        })

    intelligence_summary = {
        "summary": compliance_result.get("summary", "No summary available."),
        "category": compliance_result.get(
            "category", transcription_result.get("category", "Debt Recovery")
        ),
        "conversation_about": transcription_result.get("conversation_about", "Debt collection call"),
        "primary_intent": transcription_result.get("primary_intent", "Unknown"),
        "key_topics": transcription_result.get("key_topics", []),
        "entities": cleaned_entities,
        "root_cause": transcription_result.get("root_cause", "Unknown"),
    }

    # ---- emotional_and_tonal_analysis ----
    emotional_graph = compliance_result.get("emotional_graph", [])
    # Merge acoustic arousal if not already in emotional_graph
    if acoustic_segments and emotional_graph:
        acoustic_map = {seg["timestamp"]: seg["acoustic_arousal"] for seg in acoustic_segments}
        for point in emotional_graph:
            if "acoustic_arousal" not in point or not point["acoustic_arousal"]:
                point["acoustic_arousal"] = acoustic_map.get(point.get("timestamp", ""), "Low")

    emotional_and_tonal_analysis = {
        "overall_sentiment": compliance_result.get("overall_sentiment", "Neutral"),
        "emotional_tone": compliance_result.get("emotional_tone", "Neutral"),
        "tone_progression": compliance_result.get("tone_progression", ["Neutral"]),
        "emotional_graph": emotional_graph,
        "emotion_timeline": compliance_result.get(
            "emotion_timeline",
            [
                {"time": "start", "emotion": "neutral"},
                {"time": "middle", "emotion": "neutral"},
                {"time": "end", "emotion": "neutral"},
            ],
        ),
    }

    # ---- compliance_and_risk_audit ----
    policy_violations = compliance_result.get("policy_violations", [])

    # Normalize each violation: ensure 'severity' field exists
    _severity_values = {"low", "medium", "high", "critical"}
    for v in policy_violations:
        sev = str(v.get("severity", "")).lower()
        if sev not in _severity_values:
            v["severity"] = "medium"

    # Ensure time violation is included if detected and not already present
    if time_violation_result.get("violation", False):
        existing_ids = {v.get("clause_id") for v in policy_violations}
        if "INTERNAL-TIME-01" not in existing_ids:
            policy_violations.append(
                {
                    "clause_id": "INTERNAL-TIME-01",
                    "rule_name": time_violation_result.get("rule_name", "Operating Hours Compliance"),
                    "severity": "high",
                    "description": time_violation_result.get("description", ""),
                    "timestamp": time_violation_result.get("ist_time", "??:??"),
                    "evidence_quote": (
                        f"Call timestamp detected as {time_violation_result.get('ist_time', 'unknown')} IST."
                    ),
                }
            )

    has_violations = len(policy_violations) > 0
    is_within_policy = compliance_result.get("is_within_policy", not has_violations)

    compliance_flags = compliance_result.get("compliance_flags", [])
    if has_violations and not compliance_flags:
        compliance_flags = ["Policy Violation Detected"]

    compliance_and_risk_audit = {
        "is_within_policy": is_within_policy,
        "compliance_flags": compliance_flags,
        "policy_violations": policy_violations,
        "detected_threats": compliance_result.get("detected_threats", []),
        "risk_scores": {
            "fraud_risk": compliance_result.get("fraud_risk", "low"),
            "escalation_risk": compliance_result.get("escalation_risk", "low"),
            "urgency_level": compliance_result.get("urgency_level", "low"),
            "risk_escalation_score": compliance_result.get("risk_escalation_score", 0),
        },
    }

    # ---- performance_and_outcomes ----
    performance_and_outcomes = {
        "agent_performance": {
            "politeness": compliance_result.get("agent_politeness", "fair"),
            "empathy": compliance_result.get("agent_empathy", "medium"),
            "professionalism": compliance_result.get("agent_professionalism", "fair"),
            "overall_quality_score": compliance_result.get("agent_quality_score", 50),
        },
        "call_outcome_prediction": compliance_result.get("call_outcome_prediction", "Resolved"),
        "repeat_complaint_detected": compliance_result.get("repeat_complaint_detected", False),
        "final_status": compliance_result.get("final_status", "Pending Review"),
        "recommended_action": compliance_result.get("recommended_action", "Review manually."),
    }

    # ---- assemble final output ----
    return {
        "request_id": request_id,
        "metadata": metadata,
        "config_applied": config_applied,
        "intelligence_summary": intelligence_summary,
        "emotional_and_tonal_analysis": emotional_and_tonal_analysis,
        "compliance_and_risk_audit": compliance_and_risk_audit,
        "transcript_threads": transcript_threads,
        "performance_and_outcomes": performance_and_outcomes,
    }
