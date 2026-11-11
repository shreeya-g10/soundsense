"""Alert delivery: desktop notifications, tkinter always-on-top flash, and optional Telegram.

Owner: Vaidehi (Role 2).
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

PRIORITY_COLORS: dict[str, tuple[str, str]] = {
    "critical": ("#DC2626", "#FFFFFF"),
    "high": ("#EA580C", "#FFFFFF"),
    "medium": ("#D97706", "#111827"),
    "low": ("#0284C7", "#FFFFFF"),
}


@dataclass
class FlashPayload:
    """Visual payload passed to the tkinter flash thread."""

    title: str
    subtitle: str
    priority: str
    style: str  # "full" | "compact"
    escalated: bool


class ScreenFlasher:
    """Thread-safe always-on-top tkinter screen flash banner.

    Runs tkinter exclusively on its own daemon thread so Windows GUI thread
    affinity is preserved and FastAPI's event loop is never blocked.
    """

    def __init__(self, enabled: bool = True, duration_ms: int = 3500) -> None:
        self.enabled = enabled
        self.duration_ms = duration_ms
        self._queue: queue.Queue[FlashPayload | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._window: Any = None
        self._hide_after_id: str | None = None

    def start(self) -> None:
        """Start the tkinter thread if enabled."""
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run_tk, name="SoundSenseFlash", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Signal the tkinter mainloop to exit cleanly."""
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            self._queue.put(None)
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)

    def flash(self, alert_msg: dict[str, Any], alert_style: str = "full") -> None:
        """Queue an always-on-top visual alert flash."""
        if not self.enabled:
            return
        self.start()
        room = alert_msg.get("room", "Laptop")
        custom = alert_msg.get("custom_name")
        label = alert_msg.get("label", alert_msg.get("category", "Sound Alert"))
        title = f"{label}" if not custom else f"{label} ({custom})"
        priority = str(alert_msg.get("priority", "medium"))
        escalated = bool(alert_msg.get("escalated", False))
        prefix = "ESCALATED • " if escalated else ""
        subtitle = f"{prefix}{priority.upper()} • Room: {room} • Click or press Esc to dismiss"
        self._queue.put(
            FlashPayload(
                title=title,
                subtitle=subtitle,
                priority=priority,
                style=alert_style,
                escalated=escalated,
            )
        )

    def _run_tk(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            log.warning("tkinter unavailable; screen flash disabled")
            return

        try:
            root = tk.Tk()
        except Exception:
            log.warning("Could not initialize tkinter display; screen flash disabled")
            return

        root.withdraw()

        def hide_banner(*_args: Any) -> None:
            if self._window is not None:
                try:
                    self._window.destroy()
                except Exception:
                    pass
                self._window = None
            self._hide_after_id = None

        def show_banner(payload: FlashPayload) -> None:
            hide_banner()
            bg, fg = PRIORITY_COLORS.get(payload.priority, ("#1F2937", "#FFFFFF"))

            win = tk.Toplevel(root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.configure(bg=bg)

            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            if payload.style == "compact":
                w, h = 420, 110
                x = max(0, screen_w - w - 24)
                y = 24
            else:
                w = min(760, max(420, int(screen_w * 0.55)))
                h = 180
                x = max(0, (screen_w - w) // 2)
                y = max(24, int(screen_h * 0.08))

            win.geometry(f"{w}x{h}+{x}+{y}")

            frame = tk.Frame(win, bg=bg, padx=20, pady=16)
            frame.pack(fill="both", expand=True)

            title_lbl = tk.Label(
                frame,
                text=payload.title,
                bg=bg,
                fg=fg,
                font=("Segoe UI", 20 if payload.style == "full" else 14, "bold"),
                wraplength=w - 40,
            )
            title_lbl.pack(anchor="center", pady=(8, 6))

            sub_lbl = tk.Label(
                frame,
                text=payload.subtitle,
                bg=bg,
                fg=fg,
                font=("Segoe UI", 11),
                wraplength=w - 40,
            )
            sub_lbl.pack(anchor="center")

            for widget in (win, frame, title_lbl, sub_lbl):
                widget.bind("<Button-1>", hide_banner)
            win.bind("<Escape>", hide_banner)
            win.focus_force()

            self._window = win
            duration = self.duration_ms + (1500 if payload.escalated else 0)
            self._hide_after_id = root.after(duration, hide_banner)

        def poll() -> None:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    hide_banner()
                    root.quit()
                    return
                show_banner(item)
            root.after(100, poll)

        root.after(100, poll)
        try:
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass


class DesktopNotifierWrapper:
    """Async wrapper around desktop-notifier that degrades gracefully."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._notifier: Any = None

    def _get_notifier(self) -> Any:
        if self._notifier is None:
            from desktop_notifier import DesktopNotifier

            self._notifier = DesktopNotifier(app_name="SoundSense")
        return self._notifier

    async def send(self, alert_msg: dict[str, Any]) -> bool:
        """Send an OS desktop notification for an alert."""
        if not self.enabled:
            return False
        label = alert_msg.get("label", alert_msg.get("category", "Sound"))
        priority = str(alert_msg.get("priority", "medium")).upper()
        room = alert_msg.get("room", "Laptop")
        custom = alert_msg.get("custom_name")
        esc = " [ESCALATED]" if alert_msg.get("escalated") else ""
        title = f"SoundSense ({priority}){esc}: {label}"
        body = f"Room: {room}" + (f" • Custom: {custom}" if custom else "")
        try:
            notifier = self._get_notifier()
            await notifier.send(title=title, message=body)
            return True
        except Exception as exc:
            log.warning("Desktop notification failed: %s", exc)
            return False


class TelegramNotifier:
    """Optional free Telegram Bot API alert forwarder (off by default)."""

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s

    async def send(self, alert_msg: dict[str, Any], telegram_cfg: dict[str, Any]) -> bool:
        """Forward alert to Telegram only when explicitly enabled and configured."""
        if not telegram_cfg.get("enabled"):
            return False
        token = str(telegram_cfg.get("bot_token", "")).strip()
        chat_id = str(telegram_cfg.get("chat_id", "")).strip()
        if not token or not chat_id:
            return False

        label = alert_msg.get("label", alert_msg.get("category", "Sound"))
        priority = str(alert_msg.get("priority", "medium")).upper()
        room = alert_msg.get("room", "Laptop")
        esc = " [ESCALATED]" if alert_msg.get("escalated") else ""
        text = f"SoundSense Alert ({priority}){esc}: {label} in {room}"

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(url, json={"chat_id": chat_id, "text": text})
                return resp.status_code == 200
        except Exception as exc:
            log.warning("Telegram forwarding failed: %s", exc)
            return False


class AlertDispatcher:
    """Dispatches alerts to desktop notifications, screen flash, and Telegram."""

    def __init__(
        self,
        *,
        flasher: ScreenFlasher | None = None,
        desktop: DesktopNotifierWrapper | None = None,
        telegram: TelegramNotifier | None = None,
    ) -> None:
        self.flasher = flasher or ScreenFlasher(enabled=True)
        self.desktop = desktop or DesktopNotifierWrapper(enabled=True)
        self.telegram = telegram or TelegramNotifier()

    def start(self) -> None:
        self.flasher.start()

    def stop(self) -> None:
        self.flasher.stop()

    async def dispatch(self, alert_msg: dict[str, Any], settings: dict[str, Any]) -> None:
        """Deliver an alert across all enabled notification channels."""
        style = str(settings.get("alert_style", "full"))
        self.flasher.flash(alert_msg, alert_style=style)
        await self.desktop.send(alert_msg)
        await self.telegram.send(alert_msg, settings.get("telegram", {}))
