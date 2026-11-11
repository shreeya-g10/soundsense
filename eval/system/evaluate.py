"""End-to-end system evaluation for SoundSense backend (Role 2).

Owner: Vaidehi (Role 2).

Measures end-to-end alert latency, scenario reliability, suppression accuracy
(TV guard and quiet hours), false-alarm rate, and process CPU/RAM overhead
using FakeListener scenarios and an isolated temporary SQLite database.

Usage (from repo root):
    python -m eval.system.evaluate
    python -m eval.system.evaluate --report-dir eval/system/out
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

from contracts.types import (
    BABY_CRY,
    CATEGORIES,
    CONTEXT_MUSIC,
    CONTEXT_SPEECH,
    CONTEXT_TELEVISION,
    DOORBELL,
    EMBEDDING_SIZE,
    PRESSURE_COOKER,
    SMOKE_ALARM,
    WindowResult,
)

from backend.api.metrics import ProcessMetrics
from backend.engine.detector import DetectionEngine, EngineConfig
from backend.engine.fake_listener import SCENARIOS, FakeListener
from backend.storage.store import SQLiteStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _run_scenario(
    name: str,
    *,
    interval: float = 0.01,
    engine_config: EngineConfig | None = None,
    quiet_hours: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(Path(tmp) / f"{name}.db")
        if quiet_hours is not None:
            s = store.get_settings()
            s["quiet_hours"] = quiet_hours
            store.save_settings(s)

        latencies_ms: list[float] = []
        alerts: list[dict[str, Any]] = []
        levels: list[dict[str, Any]] = []
        window_ts_ref = [time.time()]

        def on_alert(alert_msg: dict[str, Any], _event: dict[str, Any]) -> None:
            alerts.append(alert_msg)
            latencies_ms.append(max(0.0, (time.time() - window_ts_ref[0]) * 1000.0))

        engine = DetectionEngine(
            store,
            on_level=levels.append,
            on_alert=on_alert,
            config=engine_config,
        )

        def timed_process(w: WindowResult) -> None:
            window_ts_ref[0] = w.timestamp
            engine.process_window(w)

        listener = FakeListener(
            on_window=timed_process,
            scenario=name,
            interval=interval,
            seed=42,
        )
        expected_steps = len(SCENARIOS[name]())
        listener.start()
        deadline = time.monotonic() + max(3.0, expected_steps * interval * 2.5)
        while time.monotonic() < deadline and len(levels) < expected_steps:
            time.sleep(0.01)
        listener.stop()

        history = store.get_history(limit=500)
        store.close()

    return {
        "scenario": name,
        "windows_processed": len(levels),
        "alerts_fired": len(alerts),
        "suppressed_events": sum(1 for h in history if not h["alerted"]),
        "categories_alerted": [a["category"] for a in alerts],
        "labels": [a["label"] for a in alerts],
        "escalated_count": sum(1 for a in alerts if a["escalated"]),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_latency_ms": (
            round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 2)
            if latencies_ms
            else 0.0
        ),
    }


def _eval_tv_guard_false_alarms() -> dict[str, Any]:
    """Compare false alerts on noisy TV/speech background with tv_guard on vs off."""
    results: dict[str, int] = {}
    for guard_enabled in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(Path(tmp) / f"tv_{guard_enabled}.db")
            s = store.get_settings()
            s["tv_guard"] = guard_enabled
            store.save_settings(s)

            alerts: list[dict[str, Any]] = []
            engine = DetectionEngine(
                store,
                on_alert=lambda a, _e: alerts.append(a),
                config=EngineConfig(cooldowns_s={DOORBELL: 0.01}),
            )
            t0 = 1_700_000_000.0
            for idx in range(20):
                scores = {c: 0.02 for c in CATEGORIES}
                scores[DOORBELL] = 0.61  # borderline TV false-positive spike
                engine.process_window(
                    WindowResult(
                        timestamp=t0 + idx,
                        rms=0.22,
                        skipped=False,
                        category_scores=scores,
                        context_scores={
                            CONTEXT_TELEVISION: 0.85,
                            CONTEXT_SPEECH: 0.6,
                            CONTEXT_MUSIC: 0.25,
                        },
                        embedding=[0.0] * EMBEDDING_SIZE,
                        custom_matches=[],
                        inference_ms=20.0,
                    )
                )
            store.close()
            results["tv_guard_on" if guard_enabled else "tv_guard_off"] = len(alerts)

    off_cnt = results["tv_guard_off"]
    on_cnt = results["tv_guard_on"]
    reduction = round((off_cnt - on_cnt) / off_cnt, 3) if off_cnt > 0 else 0.0
    return {
        "false_alerts_tv_guard_off": off_cnt,
        "false_alerts_tv_guard_on": on_cnt,
        "false_alert_reduction_ratio": reduction,
    }


def _eval_quiet_hours_suppression() -> dict[str, Any]:
    """Verify non-critical alerts are suppressed while critical alerts break through."""
    now_dt = datetime.now().astimezone()
    qh = {
        "enabled": True,
        "start": f"{(now_dt.hour - 1) % 24:02d}:00",
        "end": f"{(now_dt.hour + 1) % 24:02d}:00",
    }
    doorbell_res = _run_scenario(
        "doorbell_with_tv",
        interval=0.01,
        engine_config=EngineConfig(cooldowns_s={DOORBELL: 0.15}),
        quiet_hours=qh,
    )
    smoke_res = _run_scenario(
        "smoke_alarm",
        interval=0.01,
        engine_config=EngineConfig(cooldowns_s={SMOKE_ALARM: 0.5}),
        quiet_hours=qh,
    )
    return {
        "quiet_hours_doorbell_alerts": doorbell_res["alerts_fired"],
        "quiet_hours_doorbell_suppressed": doorbell_res["suppressed_events"],
        "quiet_hours_smoke_alarm_alerts": smoke_res["alerts_fired"],
    }


def run_evaluation() -> dict[str, Any]:
    """Run all Role 2 system benchmarks and return a structured report."""
    metrics = ProcessMetrics()
    _ = metrics.cpu_percent()

    results = [
        _run_scenario(
            "whistles",
            interval=0.01,
            engine_config=EngineConfig(cooldowns_s={PRESSURE_COOKER: 0.15}),
        ),
        _run_scenario(
            "smoke_alarm",
            interval=0.01,
            engine_config=EngineConfig(
                cooldowns_s={SMOKE_ALARM: 0.05},
                escalation_delays_s={"critical": 0.10},
            ),
        ),
        _run_scenario(
            "doorbell_with_tv",
            interval=0.01,
            engine_config=EngineConfig(cooldowns_s={DOORBELL: 0.15}),
        ),
        _run_scenario(
            "baby_long",
            interval=0.005,
            engine_config=EngineConfig(
                cooldowns_s={BABY_CRY: 0.10},
                escalation_delays_s={"high": 0.25},
            ),
        ),
        _run_scenario("silence", interval=0.005),
    ]

    all_latencies = [r["mean_latency_ms"] for r in results if r["alerts_fired"] > 0]
    return {
        "scenarios": results,
        "tv_guard_eval": _eval_tv_guard_false_alarms(),
        "quiet_hours_eval": _eval_quiet_hours_suppression(),
        "overall_mean_latency_ms": round(statistics.mean(all_latencies), 2) if all_latencies else 0.0,
        "cpu_percent": metrics.cpu_percent(),
        "ram_mb": metrics.ram_mb(),
    }


def write_report_artifacts(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    """Write JSON, pandas CSV summary, and matplotlib latency chart into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "system_eval.json"
    csv_path = out_dir / "system_eval.csv"
    chart_path = out_dir / "latency_chart.png"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    df = pd.DataFrame(report["scenarios"])[
        [
            "scenario",
            "windows_processed",
            "alerts_fired",
            "suppressed_events",
            "escalated_count",
            "mean_latency_ms",
            "p95_latency_ms",
        ]
    ]
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["scenario"], df["mean_latency_ms"], color="#2563EB")
    ax.set_title("SoundSense Role 2 Mean Alert Latency by Scenario")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Scenario")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    return json_path, csv_path, chart_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SoundSense Role 2 system evaluation")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON, CSV, and matplotlib chart artifacts",
    )
    args = parser.parse_args()

    report = run_evaluation()
    print("=== SoundSense System Evaluation (Role 2) ===")
    for s in report["scenarios"]:
        print(
            f"- {s['scenario']:18s} | windows={s['windows_processed']:3d} "
            f"| alerts={s['alerts_fired']:2d} | suppressed={s['suppressed_events']:2d} "
            f"| escalated={s['escalated_count']:2d} | mean_lat={s['mean_latency_ms']:.2f} ms"
        )
    tv = report["tv_guard_eval"]
    qh = report["quiet_hours_eval"]
    print(
        f"TV Guard: off={tv['false_alerts_tv_guard_off']} -> on={tv['false_alerts_tv_guard_on']} "
        f"(reduction={tv['false_alert_reduction_ratio']:.1%})"
    )
    print(
        f"Quiet Hours: doorbell_alerts={qh['quiet_hours_doorbell_alerts']} "
        f"(suppressed={qh['quiet_hours_doorbell_suppressed']}), "
        f"critical_smoke_alerts={qh['quiet_hours_smoke_alarm_alerts']}"
    )
    print(
        f"Overall mean latency: {report['overall_mean_latency_ms']:.2f} ms | "
        f"CPU: {report['cpu_percent']}% | RAM: {report['ram_mb']} MB"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.json_out}")

    if args.report_dir is not None:
        j, c, p = write_report_artifacts(report, args.report_dir)
        print(f"Saved evaluation artifacts: {j}, {c}, {p}")


if __name__ == "__main__":
    main()
