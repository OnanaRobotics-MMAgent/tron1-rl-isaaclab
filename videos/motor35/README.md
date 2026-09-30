# Motor35 videos

The three playback paths (locomotion, inverted recovery, and recovery-to-locomotion)
were rendered with `checkpoints/motor35/*/preflight_zero_policy.pt` to verify the
Motor35 task wiring, asset loading, and video path. They are not successful-policy
demonstrations. After training, render each task again using the matching weights
from `checkpoints/motor35/`.

To reproduce these interface checks, run from the repository root with `SIM_PY`
set to your Isaac Sim Python launcher. Outputs go to new timestamped directories
under `videos/motor35/`.

```bash
"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Motor35-Locomotion-Play-v0 \
  --checkpoint_path checkpoints/motor35/locomotion/preflight_zero_policy.pt \
  --num_envs 1 --seed 20260930 --velocity 0.3 0 0 --stand_seconds 2 \
  --headless --video --video_length 500 --max_steps 500

"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Motor35-Recovery-Inverted-Play-v0 \
  --checkpoint_path checkpoints/motor35/recovery/preflight_zero_policy.pt \
  --num_envs 1 --seed 20260930 \
  --headless --video --video_length 500 --max_steps 500

"$SIM_PY" scripts/rsl_rl/play_recovery_locomotion.py \
  --robot motor35 \
  --recovery_checkpoint checkpoints/motor35/recovery/preflight_zero_policy.pt \
  --locomotion_checkpoint checkpoints/motor35/locomotion/preflight_zero_policy.pt \
  --num_envs 1 --seed 20260924 --duration 10 --forward_speed 0.3 \
  --headless --video
```
