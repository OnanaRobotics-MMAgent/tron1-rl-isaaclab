# Motor43 rendered playback

These videos were rendered on 2026-09-30 from the bundled checkpoints in
`checkpoints/motor43/` and the bundled validated Motor43 asset.

- `locomotion/`: standalone locomotion playback with a zero-to-forward command.
- `recovery/`: fixed 170–180° inverted recovery playback.
- `handoff/`: recovery followed by smooth locomotion handoff.

Each directory contains `playback.json` with the task, checkpoint SHA256,
observation/action dimensions and playback counters.

Run from the repository root after setting `SIM_PY` to the Isaac Sim Python
launcher on your machine. These commands reproduce the recording settings;
they write new timestamped directories under `videos/motor43/`.

```bash
"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Motor43-Locomotion-Play-v0 --asset validated \
  --checkpoint_path checkpoints/motor43/locomotion/model_10000.pt \
  --num_envs 1 --seed 20260924 --velocity 0 0.3 0 --stand_seconds 3 \
  --headless --video --video_length 1000 --max_steps 1000

"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Motor43-Recovery-Inverted-Play-v0 --asset validated \
  --checkpoint_path checkpoints/motor43/recovery/model_30000.pt \
  --num_envs 1 --seed 20260924 \
  --headless --video --video_length 750 --max_steps 750

"$SIM_PY" scripts/rsl_rl/play_recovery_locomotion.py \
  --robot motor43 --num_envs 1 --seed 20260924 \
  --duration 15 --forward_speed 0.3 --headless --video
```

The locomotion recording lasts 20 seconds and includes the task's timed reset.
The 15-second recovery recording completed 5 successful recoveries. The
15-second handoff recording entered locomotion once, with no resets or retreats.
These counters describe these recordings, not a large-sample evaluation.
