"""
Pydantic v2 schemas for Vigilant API request/response models.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AcousticSegment(BaseModel):
    timestamp: str
    energy_score: float
    pitch_hz: float
    zcr: float
    acoustic_arousal: str  # "Low" | "Medium" | "High"


class Entity(BaseModel):
    text: str
    id: str
    type: str  # CURRENCY, ACCOUNT_TYPE, PRODUCT, PERSON


class PolicyViolation(BaseModel):
    clause_id: str
    rule_name: str
    description: str
    timestamp: str
    evidence_quote: str


class TranscriptThread(BaseModel):
    speaker: str  # "agent" | "customer"
    message: str
    timestamp: str


class EmotionalGraphPoint(BaseModel):
    timestamp: str
    tone: str
    score: float
    acoustic_arousal: str


class EmotionTimelinePoint(BaseModel):
    time: str          # "start" | "middle" | "end"
    emotion: str


class RiskScores(BaseModel):
    fraud_risk: str
    escalation_risk: str
    urgency_level: str
    risk_escalation_score: int


class AgentPerformance(BaseModel):
    politeness: str
    empathy: str
    professionalism: str
    overall_quality_score: int


# ---------------------------------------------------------------------------
# Top-level section models
# ---------------------------------------------------------------------------

class Metadata(BaseModel):
    timestamp: str
    detected_languages: list[str]
    processing_time_ms: int
    conversation_complexity: str  # "low" | "medium" | "high"


class ConfigApplied(BaseModel):
    business_domain: str
    monitored_products: list[str]
    active_policy_set: str
    risk_triggers: list[str]


class IntelligenceSummary(BaseModel):
    summary: str
    category: str
    conversation_about: str
    primary_intent: str
    key_topics: list[str]
    entities: list[Entity]
    root_cause: str


class EmotionalAnalysis(BaseModel):
    overall_sentiment: str
    emotional_tone: str
    tone_progression: list[str]
    emotional_graph: list[EmotionalGraphPoint]
    emotion_timeline: list[EmotionTimelinePoint]


class ComplianceAudit(BaseModel):
    is_within_policy: bool
    compliance_flags: list[str]
    policy_violations: list[PolicyViolation]
    detected_threats: list[str]
    risk_scores: RiskScores


class PerformanceOutcomes(BaseModel):
    agent_performance: AgentPerformance
    call_outcome_prediction: str
    repeat_complaint_detected: bool
    final_status: str
    recommended_action: str


# ---------------------------------------------------------------------------
# Root response model
# ---------------------------------------------------------------------------

class VIgilantResponse(BaseModel):
    request_id: str
    metadata: Metadata
    config_applied: ConfigApplied
    intelligence_summary: IntelligenceSummary
    emotional_and_tonal_analysis: EmotionalAnalysis
    compliance_and_risk_audit: ComplianceAudit
    transcript_threads: list[TranscriptThread]
    performance_and_outcomes: PerformanceOutcomes
