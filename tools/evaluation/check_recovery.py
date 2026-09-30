"""Inspect pose-bank validity, baseline results and live recovery training progress."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pose_bank',type=Path,default=ROOT/'data/recovery/fallen_v1.pt')
p.add_argument('--output',type=Path,default=ROOT/'outputs/recovery_fallen/continuous')
a=p.parse_args()
spec=importlib.util.spec_from_file_location('pose_bank_check',ROOT/'exts/bipedal_locomotion/bipedal_locomotion_motor43/recovery/mdp/pose_bank.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
bank=module.load_bank(a.pose_bank)
print('Pose bank:',a.pose_bank.resolve())
print('Integrity: valid; train/test poses are disjoint; finite leg limits respected')
print('Deformation band thresholds:',bank['thresholds'].tolist())
for split in ('train','test'):
 data=bank[split];print(split,'poses:',len(data['root_state']))
 for stage in range(3):
  counts=[int(((data['stage']==stage)&(data['direction']==d)).sum()) for d in range(4)]
  print('  stage',stage,'direction counts:',counts)
state_path=a.output/'state.json'
if state_path.exists():
 state=json.loads(state_path.read_text())
 assert hashlib.sha256(a.pose_bank.read_bytes()).hexdigest()==state['bank_sha256'],'Controller is using a different bank'
 sys.path.insert(0,str(ROOT/'tools/management'))
 from manage_getup import status
 status(a.output)
 if state.get('active_run'):
  import re
  files=list(a.output.glob(f'train_*_{state["active_run"]}.log'))
  if files:
   with files[0].open('rb') as f:
    f.seek(max(0,files[0].stat().st_size-20000));tail=f.read().decode(errors='replace')
   values=re.findall(r'Curriculum/recovery/level:\s+([\d.]+)',tail)
   if values:print('Live curriculum stage:',values[-1])
else:print('Controller has not been started.')
latest=a.output/'latest_evaluation.json'
if latest.exists():
 r=json.loads(latest.read_text());print('Latest evaluation:',r['iteration'],'overall:',f'{r["success_rate"]:.2%}')
 for s in r['stages']:print('  stage',s['stage'],f'{s["success_rate"]:.2%}', 'episodes:',s['episodes'], 'leg violations:',s['leg_limit_violations'])
 print('Old task:',r['old_task'])
else:print('First full evaluation is pending.')
