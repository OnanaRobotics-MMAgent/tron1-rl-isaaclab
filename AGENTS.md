# Repository organization

- `scripts/` contains task launchers: training, playback, and task data generation.
- Put tests in `tools/tests/`. Put checks, evaluations, analysis, maintenance and development utilities under appropriate `tools/` subdirectories. Do not add a root `tests/` or `test/` directory.
- Runtime task definitions and robot assets belong in the respective Motor43/Motor35 package under `exts/bipedal_locomotion/`.
- Publish selected trained policies in `checkpoints/<robot>/` and rendered demonstrations in `videos/<robot>/`. Keep their provenance and exact replay commands alongside them. Do not publish preflight weights as trained policies.
- `logs/` and `outputs/` are disposable local run products, excluded from Git. Installation metadata, Python caches, editor databases and generated analysis caches are not source files.
- Preserve the validated observation/action interfaces, rewards, curriculum, motor limits and randomization when reorganizing files.
