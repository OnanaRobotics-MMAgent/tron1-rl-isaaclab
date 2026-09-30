"""Train settled-pose recovery in bounded chunks with held-out and old-task checks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/rsl_rl'))
from continuous_getup import atomic_json, metadata


def mastered(result):
    return (len(result['stages']) == 3 and result['success_rate'] >= .90
            and result['leg_limit_violations'] == 0
            and all(s['episodes'] >= 64 and s['success_rate'] >= .90
                    and len(s['direction_stats']) == 4
                    and all(d['episodes'] >= 16 and d['success_rate'] is not None and d['success_rate'] >= .80
                            for d in s['direction_stats']) for s in result['stages']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--pose_bank', type=Path)
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--chunk_iterations', type=int, default=2000)
    p.add_argument('--num_envs', type=int, default=4096)
    p.add_argument('--child_timeout', type=int, default=14400)
    a = p.parse_args()
    if min(a.iterations,a.chunk_iterations,a.num_envs,a.child_timeout) < 1:p.error('Counts must be positive')
    out = a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Controller is already running')
    state_path=out/'state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        pid=state.get('child_pid')
        if pid and Path(f'/proc/{pid}/cmdline').exists():
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            if str(ROOT/'scripts').encode() in cmd:p.error(f'Previous child {pid} still runs')
    else:
        if a.checkpoint is None or a.pose_bank is None:p.error('New run requires --checkpoint and --pose_bank')
        checkpoint=a.checkpoint.resolve();bank=a.pose_bank.resolve()
        iteration,_=metadata(checkpoint)
        state=dict(checkpoint=str(checkpoint),iteration=iteration,stage=0,target_iteration=iteration+a.iterations,
                   initial_checkpoint=str(checkpoint),pose_bank=str(bank),bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
                   phase='evaluate',status='ready',first_training=True,mastery_streak=0)
    bank=Path(state['pose_bank'])
    if hashlib.sha256(bank.read_bytes()).hexdigest()!=state['bank_sha256']:p.error('Pose bank changed; start a new output directory')
    env=os.environ.copy()
    env['PYTHONPATH']=f'{ROOT}/exts/bipedal_locomotion:{ROOT}/rsl_rl:' + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED']='1'
    state.pop('error', None)
    def save():
        state['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S');atomic_json(state_path,state)
    def terminate(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def run(script,args,log):
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save()
            try:
                code=child.wait(timeout=a.child_timeout)
                if code:raise RuntimeError(f'{script} exited {code}; see {log}')
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:state.pop('child_pid',None);save()
    def recover():
        tag=state.get('active_run')
        if not tag:return
        candidates=list((ROOT/'logs/rsl_rl/wf_tron_1a_fallen').glob(f'*_{tag}/model_*.pt'))
        for file in sorted(candidates,key=lambda p:int(p.stem.split('_')[-1]),reverse=True):
            try:iteration,stage=metadata(file)
            except Exception:continue
            if iteration>state['iteration']:
                state.update(checkpoint=str(file.resolve()),iteration=iteration,stage=stage,first_training=False,phase='evaluate')
            break
        state.pop('active_run',None);save()
    recover();save()
    try:
        while state['status'] not in ('mastered','needs_review'):
            if (out/'STOP').exists():state['status']='paused';save();break
            if state['phase']=='evaluate':
                iteration=state['iteration'];path=out/f'eval_{iteration}.json';oldpath=out/f'old_eval_{iteration}.json'
                state['status']='evaluating';save()
                run('tools/evaluation/evaluate_fallen_poses.py',['--checkpoint',state['checkpoint'],'--pose_bank',bank,'--output',path,'--headless'],out/f'eval_{iteration}.log')
                result=json.loads(path.read_text())
                run('tools/evaluation/evaluate_getup.py',['--task','Isaac-Motor43-GetUp-Bounded-Play-v0',
                    '--checkpoint_paths',state['checkpoint'],'--stages',1,17,32,'--num_envs',128,'--episodes_per_env',4,
                    '--output',oldpath,'--headless'],out/f'old_eval_{iteration}.log')
                old=json.loads(oldpath.read_text())
                previous=[]
                if state.get('last_evaluation') and state['last_evaluation']!=str(path):
                    previous=json.loads(Path(state['last_evaluation']).read_text())['stages']
                regression=[s['stage'] for s in result['stages'] for before in previous
                            if s['stage']==before['stage'] and before['success_rate']-s['success_rate']>.15]
                old_ok=all(s['success_rate']>=.95 and s['leg_limit_violations']==0 for s in old)
                passed=mastered(result) and old_ok
                state['mastery_streak']=state['mastery_streak']+1 if passed else 0
                summary={k:v for k,v in result.items() if k!='records'}
                summary.update(iteration=iteration,training_stage=state['stage'],old_task=[dict(stage=s['stage'],success_rate=s['success_rate']) for s in old])
                atomic_json(out/'latest_evaluation.json',summary)
                with (out/'history.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
                state.update(last_evaluation=str(path),phase='train',status='ready')
                # Require two checkpoint evaluations, not a single lucky success window.
                if state['mastery_streak']>=2:state['status']='mastered'
                if regression or not old_ok:
                    state.update(status='needs_review',regression_stages=regression,old_task_passed=old_ok)
                save();print(json.dumps(summary),flush=True);continue
            if state['iteration']>=state['target_iteration']:state['status']='budget_reached';save();break
            if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Less than 5 GiB free space')
            count=min(a.chunk_iterations,state['target_iteration']-state['iteration'])
            tag='fallen_'+uuid.uuid4().hex[:12];state.update(active_run=tag,status='training');save()
            args=['--task','Isaac-Motor43-Recovery-Fallen-v0','--pose_bank',bank,'--num_envs',a.num_envs,
                  '--resume','True','--checkpoint_path',state['checkpoint'],'--max_iterations',count,
                  '--save_interval',100,'--run_name',tag,'--headless']
            if state['first_training']:args += ['--getup_stage',0]
            before=state['iteration']
            run('scripts/rsl_rl/train.py',args,out/f'train_{before}_{tag}.log');recover()
            if state['iteration']!=before+count:raise RuntimeError('Expected final checkpoint missing')
    except BaseException as error:
        recover();state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));save();raise
    print('Controller status:',state['status'],flush=True)


if __name__=='__main__':main()
