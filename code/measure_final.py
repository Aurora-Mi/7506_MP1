"""Windows-only, read-only predictor acceptance: frozen CPU FP32 test time and OS peak RSS."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
            'PagefileUsage', 'PeakPagefileUsage')]


class ProcessEntry(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]


def process_parents():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents = {}
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return parents


def peak_working_set(handle):
    # Query while the scorer is alive: post-exit counters may reset on Windows.
    # This queries the OS-maintained peak, not merely its current working set.
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(wintypes.HANDLE(int(handle)), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def wait_and_measure_tree(process):
    # Windows venv python.exe can be a redirector spawning the actual Python
    # runtime. Measure its descendants too, not just the small redirector.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = {process.pid: int(process._handle)}
    peaks = {process.pid: 0}
    try:
        while process.poll() is None:
            parents = process_parents()
            descendants = set(handles)
            while True:
                expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
                if expanded == descendants:
                    break
                descendants = expanded
            for pid in descendants - handles.keys():
                handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
                if not handle:
                    # A descendant can exit between snapshot and OpenProcess.
                    error = ctypes.get_last_error()
                    if error == 87:
                        continue
                    raise ctypes.WinError(error)
                handles[pid] = int(handle)
                peaks[pid] = 0
            for pid, handle in handles.items():
                try:
                    peaks[pid] = max(peaks[pid], peak_working_set(handle))
                except OSError:
                    if pid in parents:
                        raise
            time.sleep(0.05)
    finally:
        for pid, handle in handles.items():
            if pid != process.pid:
                kernel.CloseHandle(handle)
    # Summing per-process peaks is conservative: peaks need not be simultaneous.
    return process.returncode, sum(peaks.values()), peaks


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_scoring(label, checkpoint, repeat, output_dir, threads):
    output = output_dir / f'{label}-r{repeat}.json'
    command = [sys.executable, str(ROOT / 'evaluate.py'), '--checkpoint', str(checkpoint.resolve()),
               '--device', 'cpu', '--precision', 'fp32', '--threads', str(threads),
               '--split', 'test', '--output', str(output.resolve())]
    with (output_dir / f'{label}-r{repeat}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        code, peak_bytes, process_peaks = wait_and_measure_tree(process)
    if code:
        raise RuntimeError(f'{label} evaluation failed; inspect {output.with_suffix(".log")}')
    result = json.loads(output.read_text())
    if result['targets'] != 428405 or result['utf8_bytes'] != 1292013:
        raise ValueError('Evaluation did not score the complete fixed test split.')
    row = dict(repeat=repeat, bpb=result['bpb'], score_seconds=result['seconds'],
               peak_working_set_bytes=peak_bytes, peak_working_set_gib=peak_bytes / 1024**3,
               process_peaks_bytes=process_peaks,
               checkpoint_sha256=result['checkpoint_sha256'])
    print(json.dumps(dict(model=label, **row)), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/baseline/checkpoint.pt')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--frozen', action='store_true', help='Confirm prediction method is frozen before test.')
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('This memory-counter helper is Windows-only.')
    if not args.frozen:
        parser.error('Freeze the method first and pass --frozen. Do not tune using test results.')
    if args.repeats < 1 or args.threads < 1:
        parser.error('repeats and threads must be positive.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to avoid overwriting acceptance records.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = file_sha(args.checkpoint)
    rows = {'baseline': [], 'final': []}
    for repeat in range(1, args.repeats + 1):
        for label, checkpoint in (('baseline', args.baseline), ('final', args.checkpoint)):
            rows[label].append(run_scoring(label, checkpoint, repeat, args.output_dir, args.threads))
    if file_sha(args.checkpoint) != before:
        raise RuntimeError('Frozen checkpoint changed during acceptance.')
    baseline_median = statistics.median(row['score_seconds'] for row in rows['baseline'])
    final_median = statistics.median(row['score_seconds'] for row in rows['final'])
    ratio = final_median / baseline_median
    peak = max(row['peak_working_set_bytes'] for row in rows['final'])
    report = dict(frozen_checkpoint_sha256=before, split='test', precision='fp32', threads=args.threads,
                  python_version=platform.python_version(), os=platform.platform(),
                  repetitions=args.repeats, runs=rows, baseline_median_seconds=baseline_median,
                  final_median_seconds=final_median, median_time_ratio=ratio,
                  each_pair_time_ratios=[f['score_seconds']/b['score_seconds']
                                         for b, f in zip(rows['baseline'], rows['final'])],
                  final_peak_working_set_bytes=peak, final_peak_working_set_gib=peak/1024**3,
                  cpu_time_pass=ratio <= 5.0, peak_ram_pass=peak <= 4 * 1024**3,
                  memory_method='Conservative sum of per-process Windows PeakWorkingSetSize maxima queried every 50 ms for scorer and descendants (including venv redirector); includes load/preparation, excludes parent measurement helper.',
                  timing_method='Unchanged evaluate.py score seconds; paired fresh CPU FP32 4-thread processes; median ratio.',
                  helper_sha256=file_sha(__file__))
    (args.output_dir / 'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
