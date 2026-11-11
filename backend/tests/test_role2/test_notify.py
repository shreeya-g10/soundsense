"""Unit tests for backend/notify/notifier.py (headless/mocked)."""

from __future__ import annotations

import asyncio

import pytest

from backend.notify.notifier import (
    AlertDispatcher,
    DesktopNotifierWrapper,
    ScreenFlasher,
    TelegramNotifier,
)


def test_screen_flasher_disabled_is_safe() -> None:
    flasher = ScreenFlasher(enabled=False)
    flasher.start()
    flasher.flash({"label": "Smoke alarm", "priority": "critical", "room": "Laptop"})
    flasher.stop()


def test_desktop_notifier_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = DesktopNotifierWrapper(enabled=True)

    class BrokenNotifier:
        async def send(self, **_kwargs):
            raise RuntimeError("No notification daemon")

    monkeypatch.setattr(wrapper, "_get_notifier", lambda: BrokenNotifier())
    ok = asyncio.run(wrapper.send({"label": "Doorbell", "priority": "medium", "room": "Laptop"}))
    assert ok is False


def test_telegram_disabled_returns_false() -> None:
    tg = TelegramNotifier()
    sent = asyncio.run(
        tg.send(
            {"label": "Doorbell", "priority": "medium", "room": "Laptop"},
            {"enabled": False, "bot_token": "", "chat_id": ""},
        )
    )
    assert sent is False


def test_alert_dispatcher_routes_channels() -> None:
    calls = []

    class FakeFlasher(ScreenFlasher):
        def __init__(self):
            super().__init__(enabled=False)

        def flash(self, alert_msg, alert_style="full"):
            calls.append(("flash", alert_style, alert_msg["id"]))

    class FakeDesktop(DesktopNotifierWrapper):
        async def send(self, alert_msg):
            calls.append(("desktop", alert_msg["id"]))
            return True

    class FakeTelegram(TelegramNotifier):
        async def send(self, alert_msg, telegram_cfg):
            calls.append(("telegram", telegram_cfg["enabled"], alert_msg["id"]))
            return False

    dispatcher = AlertDispatcher(
        flasher=FakeFlasher(),
        desktop=FakeDesktop(),
        telegram=FakeTelegram(),
    )
    asyncio.run(
        dispatcher.dispatch(
            {"id": "a1", "label": "Doorbell", "priority": "medium", "room": "Laptop"},
            {"alert_style": "compact", "telegram": {"enabled": False, "bot_token": "", "chat_id": ""}},
        )
    )
    assert calls == [
        ("flash", "compact", "a1"),
        ("desktop", "a1"),
        ("telegram", False, "a1"),
    ]
