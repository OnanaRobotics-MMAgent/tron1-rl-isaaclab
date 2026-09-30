# Wheel-leg locomotion and recovery | [中文](README_cn.md)

Isaac Lab tasks for our Motor43 and Motor35 wheel-leg robots, with the repository's custom RSL-RL history encoder. Original TRON1 demonstration tasks and PF/SF assets have been removed.

## Layout

Under `exts/bipedal_locomotion/`, both `bipedal_locomotion_motor43` and `bipedal_locomotion_motor35` contain their own `assets`, `locomotion`, `recovery`, and `play` directories. Shared runner wrappers and handoff mechanics are in `bipedal_locomotion_common`. Motor35 retains its existing inheritance from the verified Motor43 base tasks/MDP.

- Motor43 current URDF: `bipedal_locomotion_motor43/assets/urdf/motor43.urdf` (formerly `WF_TRON1A.urdf`; model content unchanged).
- Motor43 source archive: `bipedal_locomotion_motor43/assets/source/轮腿总装111.zip`.
- Motor43 validated checkpoint asset/pose: `bipedal_locomotion_motor43/assets/validated/`.
- Motor35 source: `bipedal_locomotion_motor35/assets/source/Motor35_URDF/`.
- Motor35 prepared URDF: `bipedal_locomotion_motor35/assets/urdf/robot.urdf`.

The bundled validated asset replaces the handoff script's previous runtime Git archive dependency. USD internal names remain unchanged to preserve verified binary assets. No old task ID aliases are registered.

`scripts/` contains task launchers (training, playback, pose generation). Checks, tests, analysis and maintenance tools live under `tools/`, including `tools/tests/`. Selected policies and demonstrations live in `checkpoints/` and `videos/`. Training logs, temporary outputs, `*.egg-info` installation metadata and editor/Python caches are excluded from Git.

## Install and train

Clone this repository on the server, then run from its root using the Python environment with Isaac Sim and Isaac Lab installed. This checkout was validated with Isaac Sim 5.0 / Isaac Lab 2.2.1. Replace `SIM_PY` with the server's interpreter path:

```bash
SIM_PY=/home/myoukin/isaacsim/python.sh
"$SIM_PY" -m pip install -e exts/bipedal_locomotion
"$SIM_PY" -m pip install -e rsl_rl
"$SIM_PY" tools/prepare_motor35.py
"$SIM_PY" scripts/rsl_rl/train.py \
  --task Isaac-Motor35-Locomotion-v0 --num_envs 512 --headless \
  --max_iterations 20000 --save_interval 100 --run_name motor35_dr
```

Use this repository's RSL-RL, which provides `rsl_rl.runner` and the history encoder. After validating a new Motor35 locomotion policy, initialize `Isaac-Motor35-Recovery-Progressive-v0` with it using `--resume True --checkpoint_path <file> --reset_optimizer --getup_stage 0`. Do not reuse Motor43 weights on Motor35.

The main task suffixes for `Isaac-Motor43-` and `Isaac-Motor35-` are `Locomotion-v0`, `Locomotion-Play-v0`, `Recovery-Progressive-v0`, `Recovery-Inverted-Play-v0`, and `Recovery-Locomotion-Play-v0`.

Dual-policy playback:

```bash
"$SIM_PY" scripts/rsl_rl/play_recovery_locomotion.py \
  --robot motor35 --recovery_checkpoint /path/to/recovery.pt \
  --locomotion_checkpoint /path/to/locomotion.pt --forward_speed 0.3
```

For Motor43, `--robot motor43` defaults to the validated bundled asset and the weights in `checkpoints/motor43/`, so it works after cloning without old logs or Git history. Use `--current_asset` only with policies trained on the current Motor43 model.

Both policies within each model use a 28-dimensional observation, 10-frame history, 34-dimensional actor input, and 8 actions. Motor43 rolls along body-y and Motor35 along body-x. Identical dimensions do not make weights interchangeable.

Motor43 checkpoints and three rendered playback videos are included under `checkpoints/motor43/` and `videos/motor43/`. Exact replay commands are in [Motor43 videos](videos/motor43/README.md). Motor35 has only explicitly labelled preflight smoke weights/videos; it does not yet have a trained policy. `checkpoints/manifest.json` records provenance and SHA256.

Motor35 can start training and has passed engineering preflight; full policy training and handoff performance remain to be established. See [Motor35 details](docs/motor35.md), [Motor43 playback and torque logging](docs/motor43.md), and [reorganization validation](docs/reorganization.md). All physical/task settings were preserved during reorganization.

## License

Original [Apache 2.0](LICENCE) license and source attribution are retained.
