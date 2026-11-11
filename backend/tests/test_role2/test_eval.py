"""Unit test for eval/system/evaluate.py."""

from __future__ import annotations

from eval.system.evaluate import run_evaluation, write_report_artifacts


def test_system_evaluation_and_artifacts(tmp_path) -> None:
    report = run_evaluation()
    by_name = {s["scenario"]: s for s in report["scenarios"]}

    assert by_name["whistles"]["alerts_fired"] == 3
    assert by_name["silence"]["alerts_fired"] == 0
    assert report["tv_guard_eval"]["false_alerts_tv_guard_on"] == 0
    assert report["tv_guard_eval"]["false_alerts_tv_guard_off"] > 0
    assert report["quiet_hours_eval"]["quiet_hours_doorbell_alerts"] == 0
    assert report["quiet_hours_eval"]["quiet_hours_smoke_alarm_alerts"] >= 1

    json_path, csv_path, chart_path = write_report_artifacts(report, tmp_path)
    assert json_path.is_file()
    assert csv_path.is_file()
    assert chart_path.is_file() and chart_path.stat().st_size > 0
