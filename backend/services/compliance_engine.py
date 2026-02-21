"""
Compliance Engine – Agentic LLM reasoning using Gemini 1.5 Pro.
Takes transcript, acoustic data, retrieved RAG clauses and client config,
and produces the full emotional + compliance audit analysis as JSON.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import google.generativeai as genai


# ---------------------------------------------------------------------------
# System prompt for the compliance reasoner
# ---------------------------------------------------------------------------

COMPLIANCE_PROMPT_TEMPLATE = """
You are a senior RBI (Reserve Bank of India) compliance auditor AI called "Vigilant".
You specialize in auditing debt recovery calls for policy violations, emotional tone,
and agent conduct.

You are given:
1. TRANSCRIPT: A diarized call transcript (agent vs. customer turns with timestamps)
2. ACOUSTIC DATA: Per-segment audio emotion data (energy, pitch, arousal level)
3. ALL POLICY CLAUSES: The COMPLETE set of RBI/NBFC/Internal policy clauses you MUST check
4. CLIENT CONFIG: Active risk triggers and rules for this bank
5. CALL TIMESTAMP: When this call was placed

---

TRANSCRIPT:
{transcript}

---

ACOUSTIC DATA:
{acoustic}

---

ALL POLICY CLAUSES (CHECK EVERY SINGLE ONE AGAINST THE TRANSCRIPT):
{clauses}

---

CLIENT CONFIG:
{config}

---

CALL TIMESTAMP (UTC): {timestamp}
CALL TIMESTAMP (IST): {ist_time}
TIME VIOLATION DETECTED: {time_violation}
{time_violation_detail}

---

MANDATORY COMPLIANCE CHECK INSTRUCTIONS:
1. Read EVERY clause listed in "ALL POLICY CLAUSES" above.
2. For EACH clause, decide whether the agent violated it based on the transcript.
3. If a violation is found, add it to "policy_violations" with exact evidence.
4. Be strict — a missed violation is worse than a false positive in compliance auditing.
5. Pay special attention to: threats, intimidation, unauthorized visits, calls outside permitted hours,
   mentioning police/jail/legal action without basis, lack of empathy, abusive language.

Your task: Produce a comprehensive compliance audit. Return ONLY valid JSON 
(no markdown, no explanation).

The JSON must have EXACTLY these top-level keys:

{{
  "summary": "3-sentence intelligence summary of what happened",
  "category": "call category e.g. Fraud Complaint / Debt Recovery",
  "overall_sentiment": "e.g. Negative / High Tension",
  "emotional_tone": "e.g. Distressed / Aggressive",
  "tone_progression": ["ordered list tracking tone evolution"],
  "emotional_graph": [
    {{
      "timestamp": "MM:SS",
      "tone": "Neutral|Frustrated|Angry|Threatening|Distressed|Aggressive",
      "score": 0.0,
      "acoustic_arousal": "Low|Medium|High"
    }}
  ],
  "emotion_timeline": [
    {{"time": "start", "emotion": "neutral"}},
    {{"time": "middle", "emotion": "frustrated"}},
    {{"time": "end", "emotion": "angry"}}
  ],
  "is_within_policy": false,
  "compliance_flags": ["list of high-level flag names"],
  "policy_violations": [
    {{
      "clause_id": "RBI-REC-04",
      "rule_name": "No Physical Threats",
      "severity": "low|medium|high|critical",
      "description": "explanation of what violated this clause",
      "timestamp": "MM:SS",
      "evidence_quote": "exact agent quote from transcript"
    }}
  ],
  "detected_threats": ["plain English threat descriptions"],
  "fraud_risk": "low|medium|high",
  "escalation_risk": "low|medium|high",
  "urgency_level": "low|medium|high",
  "risk_escalation_score": 0,
  "agent_politeness": "excellent|good|fair|poor|unacceptable",
  "agent_empathy": "high|medium|low|none",
  "agent_professionalism": "excellent|good|fair|poor|unacceptable",
  "agent_quality_score": 0,
  "call_outcome_prediction": "e.g. Escalation Likely / Legal Dispute",
  "repeat_complaint_detected": false,
  "final_status": "e.g. Escalated to Compliance Manager",
  "recommended_action": "specific action for compliance team"
}}

Rules:
- emotional_graph must have one entry per ~30 seconds of conversation (use transcript timestamps)
- Merge acoustic_arousal from ACOUSTIC DATA with conversational tone from transcript
- policy_violations must cite real clause_ids from the ALL POLICY CLAUSES section
- severity must be: "critical" (criminal threats/violence), "high" (intimidation/harassment), "medium" (procedural breach), "low" (minor/technical)
- If time violation was detected, add it as a policy_violation with clause_id INTERNAL-TIME-01 and severity "high"
- risk_escalation_score: 0–100 integer reflecting combined risk (consider violations, arousal, threats)
- agent_quality_score: 0–100 (100 = perfect agent, 0 = completely non-compliant)
- evidence_quote must be the exact agent utterance from the transcript
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_transcript(transcript_threads: list[dict]) -> str:
    lines = []
    for t in transcript_threads:
        speaker = t.get("speaker", "unknown").upper()
        ts = t.get("timestamp", "??:??")
        msg = t.get("message", "")
        lines.append(f"[{ts}] {speaker}: {msg}")
    return "\n".join(lines)


def _format_acoustic(acoustic_segments: list[dict]) -> str:
    lines = []
    for seg in acoustic_segments:
        lines.append(
            f"[{seg['timestamp']}] Energy={seg['energy_score']:.2f} "
            f"Pitch={seg['pitch_hz']:.0f}Hz ZCR={seg['zcr']:.4f} "
            f"Arousal={seg['acoustic_arousal']}"
        )
    return "\n".join(lines) if lines else "No acoustic data available."


def _format_clauses(clauses: list[dict]) -> str:
    if not clauses:
        return "No specific clauses retrieved. Apply general RBI recovery guidelines."
    lines = []
    for c in clauses:
        lines.append(
            f"[{c['clause_id']}] {c['rule_name']}\n  {c['description'][:200]}"
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def _build_fallback_compliance(error: str = "") -> dict:
    return {
        "summary": f"Analysis could not be completed. Error: {error}" if error else "Analysis could not be completed due to a processing error.",
        "category": "Unknown",
        "overall_sentiment": "Unknown",
        "emotional_tone": "Unknown",
        "tone_progression": ["Unknown"],
        "emotional_graph": [
            {"timestamp": "00:00", "tone": "Neutral", "score": 0.5, "acoustic_arousal": "Low"}
        ],
        "emotion_timeline": [
            {"time": "start", "emotion": "unknown"},
            {"time": "middle", "emotion": "unknown"},
            {"time": "end", "emotion": "unknown"},
        ],
        "is_within_policy": True,
        "compliance_flags": [],
        "policy_violations": [],
        "detected_threats": [],
        "fraud_risk": "low",
        "escalation_risk": "low",
        "urgency_level": "low",
        "risk_escalation_score": 0,
        "agent_politeness": "fair",
        "agent_empathy": "medium",
        "agent_professionalism": "fair",
        "agent_quality_score": 50,
        "call_outcome_prediction": "Resolved",
        "repeat_complaint_detected": False,
        "final_status": "Pending Review",
        "recommended_action": "Manual review required.",
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_compliance_analysis(
    transcript_threads: list[dict],
    acoustic_segments: list[dict],
    retrieved_clauses: list[dict],
    client_config: dict,
    call_timestamp_utc: str,
    time_violation_result: dict,
    api_key: str,
) -> dict:
    """
    Run the agentic LLM compliance reasoner.

    Returns a dict with all compliance, emotional, and performance fields.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

    # Format inputs
    transcript_text = _format_transcript(transcript_threads)
    acoustic_text = _format_acoustic(acoustic_segments)
    clauses_text = _format_clauses(retrieved_clauses)
    config_text = json.dumps(client_config, indent=2)

    ist_time = time_violation_result.get("ist_time", "unknown")
    time_viol = time_violation_result.get("violation", False)
    time_viol_detail = (
        f"TIME VIOLATION DETAIL: {time_violation_result['description']}"
        if time_viol
        else ""
    )

    prompt = COMPLIANCE_PROMPT_TEMPLATE.format(
        transcript=transcript_text,
        acoustic=acoustic_text,
        clauses=clauses_text,
        config=config_text,
        timestamp=call_timestamp_utc,
        ist_time=ist_time,
        time_violation="YES" if time_viol else "NO",
        time_violation_detail=time_viol_detail,
    )

    try:
        print("[ComplianceEngine] Running Gemini compliance analysis...")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.05,
            ),
        )
        result = _extract_json(response.text)
        print(
            f"[ComplianceEngine] Done. Violations: {len(result.get('policy_violations', []))} | "
            f"Score: {result.get('risk_escalation_score', 'N/A')}"
        )
        return result
    except json.JSONDecodeError as exc:
        print(f"[ComplianceEngine] JSON parse error: {exc}")
        return _build_fallback_compliance(f"JSON parse error: {exc}")
    except Exception as exc:
        print(f"[ComplianceEngine] ERROR with gemini-2.5-flash: {exc}")
        # Retry with gemini-2.0-flash as fallback
        try:
            print("[ComplianceEngine] Retrying with gemini-2.0-flash...")
            fallback_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
            response = fallback_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.05,
                ),
            )
            result = _extract_json(response.text)
            print(
                f"[ComplianceEngine] 2.0-flash fallback succeeded. "
                f"Violations: {len(result.get('policy_violations', []))} | "
                f"Score: {result.get('risk_escalation_score', 'N/A')}"
            )
            return result
        except json.JSONDecodeError as exc2:
            print(f"[ComplianceEngine] Fallback JSON parse error: {exc2}")
            return _build_fallback_compliance(f"JSON parse error: {exc2}")
        except Exception as exc2:
            print(f"[ComplianceEngine] All fallbacks failed: {exc2}")
            return _build_fallback_compliance(str(exc2))
