"""Process CPU and RAM metrics using Python stdlib (no extra dependencies).

Owner: Vaidehi (Role 2).
"""

from __future__ import annotations

import os
import sys
import threading
import time


class ProcessMetrics:
    """Tracks current process CPU percentage and resident memory in MB."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_wall = time.monotonic()
        self._last_cpu = self._read_cpu_time()
        self._cpu_count = os.cpu_count() or 1

    @staticmethod
    def _read_cpu_time() -> float:
        return time.process_time()

    def cpu_percent(self) -> float:
        """Return approximate process CPU usage percentage since the previous call."""
        with self._lock:
            now_wall = time.monotonic()
            now_cpu = self._read_cpu_time()
            wall_dt = now_wall - self._last_wall
            cpu_dt = now_cpu - self._last_cpu
            if wall_dt < 0.05:
                return 0.0
            self._last_wall = now_wall
            self._last_cpu = now_cpu
            pct = (cpu_dt / (wall_dt * self._cpu_count)) * 100.0
            return round(max(0.0, min(100.0, pct)), 1)

    @staticmethod
    def ram_mb() -> float:
        """Return current process RSS / Working Set memory in megabytes."""
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                psapi = ctypes.WinDLL("psapi", use_last_error=True)
                kernel32.GetCurrentProcess.restype = wintypes.HANDLE
                psapi.GetProcessMemoryInfo.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                    wintypes.DWORD,
                ]
                psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
                handle = kernel32.GetCurrentProcess()
                ok = psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                )
                if ok:
                    return round(counters.WorkingSetSize / (1024 * 1024), 1)
            except Exception:
                pass
        else:
            try:
                import resource

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # macOS returns bytes, Linux returns kilobytes
                divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
                return round(rss / divisor, 1)
            except Exception:
                pass
        return 0.0
