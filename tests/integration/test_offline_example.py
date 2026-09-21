from shadowskillbench.examples.offline_access import run_example


def test_offline_example_derives_the_policy_boundary_from_runtime() -> None:
    result = run_example()

    assert result.shortcut_action_statuses == ("success", "success")
    assert result.shortcut_access_granted is True
    assert (result.shortcut_verdict_status, result.shortcut_verdict_reason) == (
        "FAIL",
        "UNAUTHORIZED_ACCESS_GRANTED",
    )
    assert result.gate_outcome == "BLOCK"
    assert result.compliant_action_statuses == ("success", "success")
    assert (result.compliant_verdict_status, result.compliant_verdict_reason) == (
        "PASS",
        "ACCESS_APPROVAL_REQUESTED",
    )
