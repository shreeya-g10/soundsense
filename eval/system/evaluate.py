"""End-to-end system evaluation for SoundSense backend (Role 2).

Owner: Vaidehi (Role 2).

Measures end-to-end alert latency, scenario reliability, suppression accuracy,
false-alarm rate, and process CPU/RAM overhead using FakeListener scenarios
and an isolated temporary SQLite database.

Usage (from repo root):
    python -m eval.system.evaluate
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from contracts.types import BABY_CRY, DOORBELL, PRESSURE_COOKER, SMOKE_ALARM

from backend.api.metrics import ProcessMetrics
from backend.engine.detector import DetectionEngine, EngineConfig
from backend.engine.fake_listener import FakeListener
from backend.storage.store import SQLiteStore


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

        def on_alert(alert_msg: dict[str, Any], _event: dict[str, Any]) -> None:
            alerts.append(alert_msg)
            latencies_ms.append(max(0.0, (time.time() - window_ts_ref[0]) * 1000.0))

        window_ts_ref = [time.time()]
        engine = DetectionEngine(
            store,
            on_level=levels.append,
            on_alert=on_alert,
            config=engine_config,
        )

        # Wrap on_window to capture window emission timestamp synchronously
        def timed_process(w):
            window_ts_ref[0] = w.timestamp
            engine.process_window(w)

        listener = FakeListener(
            on_window=timed_process,
            scenario=name,
            interval=interval,
            seed=42,
        )
        # Step count per scenario by running until silence settles
        from backend.engine.fake_listener import SCENARIOS

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


def run_evaluation() -> dict[str, Any]:
    """Run all Role 2 system benchmarks and return a structured report."""
    metrics = ProcessMetrics()
    _ = metrics.cpu_percent()  # prime baseline

    results = [
        _run_scenario(
            "whistles",
            interval=0.01,
            # Shorten cooldown proportionally to 10ms test interval? Keep real timestamps:
            # FakeListener uses real wall clock, so use 0.05s interval -> 17 steps = 0.85s gap
            # or test with small cooldown for fast eval benchmark
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
    report = {
        "scenarios": results,
        "overall_mean_latency_ms": round(statistics.mean(all_latencies), 2) if all_latencies else 0.0,
        "cpu_percent": metrics.cpu_percent(),
        "ram_mb": metrics.ram_mb(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SoundSense Role 2 system evaluation")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = run_evaluation()
    print("=== SoundSense System Evaluation (Role 2) ===")
    for s in report["scenarios"]:
        print(
            f"- {s['scenario']:18s} | windows={s['windows_processed']:3d} "
            f"| alerts={s['alerts_fired']:2d} | suppressed={s['suppressed_events']:2d} "
            f"| escalated={s['escalated_count']:2d} | mean_lat={s['mean_latency_ms']:.2f} ms"
        )
    print(
        f"Overall mean latency: {report['overall_mean_latency_ms']:.2f} ms | "
        f"CPU: {report['cpu_percent']}% | RAM: {report['ram_mb']} MB"
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
