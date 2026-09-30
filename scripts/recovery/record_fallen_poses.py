"""Record a fixed-duration video cycling through distinct held-out fallen poses."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'exts/bipedal_locomotion'), str(ROOT/'rsl_rl')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--pose_bank',required=True)
p.add_argument('--output',required=True)
p.add_argument('--seconds',type=float,default=60)
p.add_argument('--seed',type=int,default=456)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
output=Path(a.output).resolve()
if output.exists():p.error(f'Output already exists: {output}')
if a.seconds<=0:p.error('--seconds must be positive')
a.enable_cameras=True
app=AppLauncher(a).app
import json
import random
import torch
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import bipedal_locomotion_motor43, bipedal_locomotion_motor35
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from bipedal_locomotion_common.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import OnPolicyRunner

task='Isaac-Motor43-Recovery-Fallen-Play-v0'
cfg=parse_env_cfg(task,device=a.device,num_envs=1)
cfg.fallen.bank_path=a.pose_bank;cfg.seed=a.seed
raw=gym.make(task,cfg=cfg,render_mode='rgb_array')
env=RslRlVecEnvWrapper(raw)
runner=OnPolicyRunner(env,load_cfg_from_registry(task,'rsl_rl_cfg_entry_point').to_dict(),device=raw.unwrapped.device)
runner.load(a.checkpoint);policy=runner.get_inference_policy();encoder=runner.get_inference_encoder()
bank=raw.unwrapped._fallen_bank
rng=random.Random(a.seed)
# Rotate all 12 stage/direction groups; sample without replacement within each.
groups=[]
for direction in range(4):
 for stage in range(3):
  ids=((bank['stage']==stage)&(bank['direction']==direction)).nonzero().flatten().tolist()
  rng.shuffle(ids);groups.append(ids)
sequence=[]
while any(groups):
 for group in groups:
  if group:sequence.append(group.pop())
index=0;records=[];current=None

def reset_pose(frame):
 global index,current
 if index>=len(sequence):raise RuntimeError('All held-out poses exhausted')
 pose=sequence[index];index+=1
 raw.unwrapped._fallen_forced_indices=torch.tensor([pose],device=raw.unwrapped.device)
 obs,info=env.reset();raw.unwrapped._fallen_forced_indices=None
 current=dict(pose_id=pose,stage=int(bank['stage'][pose]),direction=int(bank['direction'][pose]),start_frame=frame)
 records.append(current)
 return obs,info

fps=round(1/raw.unwrapped.step_dt);frames=round(a.seconds*fps)
output.parent.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
writer=imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart'])
try:
 with torch.inference_mode():
  obs,info=reset_pose(0)
  # Fill render products before encoding, without stepping physics or the policy.
  for _ in range(5):raw.render()
  for frame in range(frames):
   rgb=raw.render();pic=Image.fromarray(rgb);draw=ImageDraw.Draw(pic)
   text=f'{Path(a.checkpoint).stem} | Test pose {current["pose_id"]} | Joint band {current["stage"]+1}/3 | {frame/fps:05.2f}s'
   draw.rectangle((0,0,pic.width,36),fill=(18,22,28));draw.text((12,6),text,font=font,fill=(245,245,245))
   writer.append_data(np.asarray(pic))
   x=torch.cat((encoder(info['observations']['obsHistory'].flatten(1)),obs,info['observations']['commands']),-1)
   obs,_,done,info=env.step(policy(x))
   if bool(done[0]):
    current.update(end_frame=frame+1,success=bool(raw.unwrapped.termination_manager.get_term('success')[0]))
    print('EPISODE',json.dumps(current),flush=True)
    if frame+1<frames:obs,info=reset_pose(frame+1)
   if (frame+1)%500==0:print('FRAMES',frame+1,'/',frames,flush=True)
 result=dict(checkpoint=str(Path(a.checkpoint).resolve()),pose_bank=str(Path(a.pose_bank).resolve()),split='test',seed=a.seed,
             fps=fps,frames=frames,seconds=frames/fps,episodes=records)
 output.with_suffix('.json').write_text(json.dumps(result,indent=2))
 print('VIDEO',str(output),flush=True)
finally:
 writer.close();env.close();app.close()
