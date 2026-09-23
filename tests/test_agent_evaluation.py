from fertility_risk.agent_evaluation import evaluate_planner


def test_offline_agent_evaluation_gate_passes():
    report = evaluate_planner()

    assert report["passed"] is True
    assert report["intent_accuracy"] == 1.0
    assert report["tool_alignment"] is True
    assert report["safe_unknown_fallback"] is True
