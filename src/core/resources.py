"""Low-overhead process resource measurements (not algorithm-only memory)."""

import os
import sys
import threading


def resident_bytes(pid=None):
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        handle = (
            kernel.GetCurrentProcess()
            if pid is None
            else kernel.OpenProcess(0x410, False, pid)
        )
        try:
            if not handle or not get_memory(
                handle, ctypes.byref(counters), counters.cb
            ):
                raise OSError("GetProcessMemoryInfo failed")
        finally:
            if pid is not None and handle:
                kernel.CloseHandle(handle)
        return int(counters.WorkingSetSize)
    if sys.platform.startswith("linux"):
        with open(
            f"/proc/{'self' if pid is None else pid}/statm", encoding="ascii"
        ) as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    raise OSError("RSS sampler supports Windows and Linux")


class ResourceSampler:
    """Sample RSS every 10 ms; transient peaks between samples may be missed.

    Baseline and peak include the interpreter and accumulated report data.
    They must not be interpreted as isolated optimizer allocations.
    """

    def __init__(self):
        self.baseline = self.peak = resident_bytes()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, resident_bytes())

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, resident_bytes())
        return {
            "rss_baseline_bytes": self.baseline,
            "sampled_peak_rss_bytes": self.peak,
            "sampled_rss_increase_bytes": max(0, self.peak - self.baseline),
        }


class SeriesResourceSampler:
    """Peak of synchronously sampled RSS sums, not sum of process peaks."""

    def __init__(self):
        self.pids = set()
        self.lock = threading.Lock()
        self.peak = resident_bytes()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def add(self, pid):
        with self.lock:
            self.pids.add(pid)

    def discard(self, pid):
        with self.lock:
            self.pids.discard(pid)

    def _sample(self):
        while not self.stop.wait(0.01):
            with self.lock:
                pids = tuple(self.pids)
            total = resident_bytes()
            for pid in pids:
                try:
                    total += resident_bytes(pid)
                except OSError:
                    pass  # A child may exit between enumeration and sampling.
            self.peak = max(self.peak, total)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return self.peak
