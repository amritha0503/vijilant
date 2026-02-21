"""
Test script for Vigilant backend API.
Run with: python test_api.py
Requires: httpx, a test audio file at ./test_audio.mp3
"""
import json
import sys
import httpx

BASE_URL = "http://localhost:8000"

REQUIRED_KEYS = [
    "request_id",
    "metadata",
    "config_applied",
    "intelligence_summary",
    "emotional_and_tonal_analysis",
    "compliance_and_risk_audit",
    "transcript_threads",
    "performance_and_outcomes",
]


def test_health():
    print("Testing /health endpoint...")
    resp = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok", f"Unexpected status: {data}"
    print("  ✅ /health OK")


def test_analyze_default_config(audio_path: str):
    print(f"\nTesting /analyze with audio: {audio_path}")
    with open(audio_path, "rb") as f:
        files = {"audio_file": (audio_path, f, "audio/mpeg")}
        resp = httpx.post(f"{BASE_URL}/analyze", files=files, timeout=180)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    data = resp.json()

    # Check all required top-level keys
    for key in REQUIRED_KEYS:
        assert key in data, f"Missing key: {key}"
        print(f"  ✅ {key}: present")

    # Check specific nested fields
    assert "transcript_threads" in data
    assert isinstance(data["transcript_threads"], list)
    assert "policy_violations" in data["compliance_and_risk_audit"]
    assert "emotional_graph" in data["emotional_and_tonal_analysis"]
    assert "risk_scores" in data["compliance_and_risk_audit"]

    print(f"\n  📊 Summary: {data['intelligence_summary']['summary'][:100]}...")
    print(f"  🔴 Violations: {len(data['compliance_and_risk_audit']['policy_violations'])}")
    print(f"  🎭 Sentiment: {data['emotional_and_tonal_analysis']['overall_sentiment']}")
    print(f"  ⚠️  Risk Score: {data['compliance_and_risk_audit']['risk_scores']['risk_escalation_score']}")
    print(f"  ✅ /analyze (default config) PASSED")

    return data


def test_analyze_custom_config(audio_path: str):
    print(f"\nTesting /analyze with custom client config...")
    custom_config = {
        "business_domain": "Banking / Personal Loan",
        "monitored_products": ["Personal Loan"],
        "active_policy_set": "CUSTOM_INTERNAL_v1",
        "risk_triggers": ["Family Mention", "Jail Mention"],
        "custom_rules": [
            {
                "rule_id": "CUSTOM-01",
                "rule_name": "No Family Contact",
                "description": "Agent must not mention or contact family members.",
            }
        ],
    }

    with open(audio_path, "rb") as f:
        files = {
            "audio_file": (audio_path, f, "audio/mpeg"),
        }
        data_fields = {"client_config": ("config.json", json.dumps(custom_config), "application/json")}
        resp = httpx.post(f"{BASE_URL}/analyze", files={**files, **data_fields}, timeout=180)

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    assert data["config_applied"]["active_policy_set"] == "CUSTOM_INTERNAL_v1"
    print(f"  ✅ Custom config applied: {data['config_applied']['active_policy_set']}")
    print(f"  ✅ /analyze (custom config) PASSED")


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio.mp3"

    print("=" * 60)
    print("  VIGILANT API TEST SUITE")
    print("=" * 60)

    test_health()
    test_analyze_default_config(audio_file)
    test_analyze_custom_config(audio_file)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)
