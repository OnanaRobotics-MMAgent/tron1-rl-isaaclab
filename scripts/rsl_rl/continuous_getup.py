"""Run resumable GetUp training/evaluation cycles on one GPU, without a GUI."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import uuid

import torch

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def metadata(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    return int(data['iter']), int(data['task_state']['level'])


def passed(results):
    return bool(results) and all(r['success_rate'] >= 0.80 and len(r['direction_stats']) == 4
        and all(d['episodes'] >= 32 and d['success_rate'] is not None
                and d['success_rate'] >= 0.70 for d in r['direction_stats'])
        and r['leg_limit_violations'] == 0 for r in results)


def regressed_stages(previous, current):
    old_rates = {r['stage']: r['success_rate'] for r in previous}
    return [r['stage'] for r in current if r['stage'] in old_rates
            and old_rates[r['stage']] - r['success_rate'] > 0.15]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, help='Initial checkpoint; ignored when state.json already exists')
    p.add_argument('--output', type=Path, required=True, help='Persistent controller directory; reuse to resume')
    p.add_argument('--iterations', type=int, default=100000, help='Total additional updates for a new controller')
    p.add_argument('--chunk_iterations', type=int, default=2000)
    p.add_argument('--num_envs', type=int, default=4096)
    p.add_argument('--eval_envs', type=int, default=128)
    p.add_argument('--eval_episodes', type=int, default=4)
    p.add_argument('--save_interval', type=int, default=100)
    p.add_argument('--child_timeout', type=int, default=14400, help='Seconds allowed per train/evaluation process')
    a = p.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    if min(a.iterations, a.chunk_iterations, a.num_envs, a.eval_envs, a.eval_episodes, a.save_interval, a.child_timeout) < 1:
        p.error('All numeric options must be positive')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('This controller is already running')
    state_path = out / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        child_pid = state.get('child_pid')
        if child_pid and Path(f'/proc/{child_pid}/cmdline').exists():
            command = Path(f'/proc/{child_pid}/cmdline').read_bytes()
            if str(ROOT / 'scripts/rsl_rl').encode() in command:
                p.error(f'Previous child process {child_pid} is still running; stop it before resuming')
    else:
        if a.checkpoint is None:
            p.error('--checkpoint is required on the first run')
        checkpoint = a.checkpoint.resolve()
        iteration, stage = metadata(checkpoint)
        state = dict(checkpoint=str(checkpoint), iteration=iteration, stage=stage,
                     target_iteration=iteration + a.iterations, phase='evaluate', status='ready')
        atomic_json(state_path, state)
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{ROOT / 'exts/bipedal_locomotion'}:{ROOT / 'rsl_rl'}:" + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'

    def save():
        state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        atomic_json(state_path, state)

    def run(script, args, log):
        base = ROOT / ('tools/evaluation' if script == 'evaluate_getup.py' else 'scripts/rsl_rl')
        cmd = [sys.executable, str(base / script), *map(str, args)]
        print('Running:', ' '.join(cmd), '\nLog:', log, flush=True)
        with log.open('w') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            state['child_pid'] = child.pid
            save()
            try:
                code = child.wait(timeout=a.child_timeout)
                if code:
                    raise RuntimeError(f'{script} exited {code}; see {log}')
            except BaseException:
                import signal
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise
            finally:
                state.pop('child_pid', None)
                save()

    def recover_training():
        tag = state.get('active_run')
        if not tag:
            return
        candidates = list((ROOT / 'logs/rsl_rl/wf_tron_1a_getup_bounded').glob(f'*_{tag}/model_*.pt'))
        for path in sorted(candidates, key=lambda p: int(p.stem.split('_')[1]), reverse=True):
            try:
                iteration, stage = metadata(path)
            except Exception:
                continue
            if iteration > state['iteration']:
                state.update(checkpoint=str(path.resolve()), iteration=iteration, stage=stage, phase='evaluate')
            break
        state.pop('active_run', None)
        save()

    recover_training()
    try:
        while state.get('status') not in ('mastered', 'needs_review'):
            if (out / 'STOP').exists():
                state['status'] = 'paused'; save(); break
            if state['phase'] == 'evaluate':
                stage = state['stage']
                stages = list(range(33)) if stage == 32 else sorted({1, max(0, stage - 1), stage})
                result_path = out / f"eval_{state['iteration']}.json"
                state['status'] = 'evaluating'; save()
                run('evaluate_getup.py', ['--task', 'Isaac-Motor43-GetUp-Bounded-Play-v0',
                    '--checkpoint_paths', state['checkpoint'], '--stages', *stages,
                    '--num_envs', a.eval_envs, '--episodes_per_env', a.eval_episodes,
                    '--output', result_path, '--headless'], out / f"eval_{state['iteration']}.log")
                results = json.loads(result_path.read_text())
                previous_results = []
                previous_path = state.get('last_evaluation')
                if previous_path and Path(previous_path) != result_path:
                    previous_results = json.loads(Path(previous_path).read_text())
                regressions = regressed_stages(previous_results, results)
                summary = dict(iteration=state['iteration'], stage=stage, checkpoint=state['checkpoint'],
                               evaluations=[{k: r[k] for k in ('stage', 'tilt_range', 'success_rate',
                                   'direction_stats', 'mean_action_excess', 'leg_limit_violations')} for r in results])
                with (out / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(summary) + '\n')
                atomic_json(out / 'latest_evaluation.json', summary)
                state.update(phase='train', status='ready', last_evaluation=str(result_path))
                if stage == 32 and len(results) == 33 and passed(results):
                    state['status'] = 'mastered'
                if regressions:
                    state.update(status='needs_review', regression_stages=regressions,
                                 previous_evaluation=previous_path)
                save()
                print(json.dumps(summary), flush=True)
                continue
            if state['iteration'] >= state['target_iteration']:
                state['status'] = 'budget_reached'; save(); break
            count = min(a.chunk_iterations, state['target_iteration'] - state['iteration'])
            import shutil
            if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
                raise RuntimeError('Less than 5 GiB free space; paused before starting another training segment')
            tag = 'continuous_' + uuid.uuid4().hex[:12]
            state.update(active_run=tag, status='training'); save()
            run('train.py', ['--task', 'Isaac-Motor43-GetUp-Bounded-v0', '--num_envs', a.num_envs,
                '--resume', 'True', '--checkpoint_path', state['checkpoint'], '--max_iterations', count,
                '--save_interval', a.save_interval, '--run_name', tag, '--headless'],
                out / f"train_from_{state['iteration']}_{tag}.log")
            previous = state['iteration']
            recover_training()
            if state['iteration'] != previous + count:
                raise RuntimeError('Training did not produce the expected final checkpoint')
    except BaseException as error:
        recover_training()
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save()
        raise
    print('Controller status:', state['status'], '\nState:', state_path, flush=True)


if __name__ == '__main__':
    main()
