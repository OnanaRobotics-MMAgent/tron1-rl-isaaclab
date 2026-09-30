# Motor35 checkpoints

There are no fully trained Motor35 policies in this repository yet. The two
`preflight_zero_policy.pt` files are tiny interface smoke weights used only to
render the task-path videos; they must not be used as a locomotion or recovery
policy claim.

After training and validation, copy the full checkpoints to:

- `checkpoints/motor35/locomotion/model_<iteration>.pt`
- `checkpoints/motor35/recovery/model_<iteration>.pt`

Both training and playback use the Motor35 model, motor limits and domain
randomization settings already defined in its task package. Do not use Motor43
weights here. Training commands are in `README_cn.md`.
