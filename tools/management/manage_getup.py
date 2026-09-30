"""Idempotent launcher and live status for the continuous GetUp controller."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def locked(directory):
    with (directory / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def status(directory):
    active = locked(directory)
    print('Controller:', 'RUNNING' if active else 'STOPPED')
    path = directory / 'state.json'
    if path.exists():
        state = json.loads(path.read_text())
        print('Phase:', state['status'], 'Stage:', state['stage'])
        print('Last saved checkpoint:', state['checkpoint'])
        if state.get('active_run'):
            logs = list(directory.glob(f"train_*_{state['active_run']}.log"))
            if logs:
                with logs[0].open('rb') as stream:
                    stream.seek(max(0, logs[0].stat().st_size - 16000))
                    text = stream.read().decode(errors='replace')
                iterations = re.findall(r'Learning iteration (\d+)/(\d+)', text)
                if iterations:
                    print('Live learning iteration:', '/'.join(iterations[-1]))
                print('Live training log:', logs[0])
        if state.get('error'):
            print('Last error:', state['error'])
    print('Console:', directory / 'console.log')
    return active


def main(controller_path=None, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'stop'])
    parser.add_argument('--output', type=Path, default=default_output or ROOT / 'logs/getup_continuous')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pose_bank', type=Path)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--chunk_iterations', type=int, default=2000)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.command == 'status':
        status(output); return
    if args.command == 'stop':
        (output / 'STOP').touch()
        print('Pause requested. Current training/evaluation segment will finish first.')
        status(output); return
    with (output / '.launch.lock').open('a') as startup_lock:
        fcntl.flock(startup_lock, fcntl.LOCK_EX)
        if locked(output):
            print('Already running; no second process started and no logs overwritten.')
            status(output); return
        if not (output / 'state.json').exists() and args.checkpoint is None:
            parser.error('--checkpoint is required for a new output directory')
        (output / 'STOP').unlink(missing_ok=True)
        command = [str(args.python), str(controller_path or ROOT / 'scripts/rsl_rl/continuous_getup.py'),
                   '--output', str(output), '--iterations', str(args.iterations),
                   '--chunk_iterations', str(args.chunk_iterations), '--num_envs', str(args.num_envs)]
        if args.checkpoint is not None:
            command += ['--checkpoint', str(args.checkpoint.resolve())]
        if args.pose_bank is not None:
            command += ['--pose_bank', str(args.pose_bank.resolve())]
        with (output / 'console.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if locked(output):
                status(output); return
            if child.poll() is not None:
                raise SystemExit(f'Controller exited {child.returncode}. See {output / "console.log"}')
            time.sleep(.1)
        print('Controller is starting; PID:', child.pid)
        status(output)


if __name__ == '__main__':
    main()
