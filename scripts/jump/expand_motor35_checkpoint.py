"""Expand a Motor35 flat checkpoint for the three appended jump observations.

Usage: python scripts/jump/expand_motor35_checkpoint.py model_10000.pt jump_init.pt
The original checkpoint is never modified. Use --reset_optimizer when training.
"""

import argparse
from pathlib import Path

import torch


OLD_OBS = 28
NEW_OBS = 31
COMMANDS = 3
LATENT = 3
HISTORY = 10


def expand_checkpoint(checkpoint):
    model = checkpoint["model_state_dict"]
    encoder = checkpoint["encoder_state_dict"]
    actor = model["actor.0.weight"]
    critic = model["critic.0.weight"]
    history = encoder["encoder.0.weight"]
    if actor.shape[1] != LATENT + OLD_OBS + COMMANDS or model["actor.6.weight"].shape[0] != 8:
        raise ValueError(f"Unexpected flat actor input: {actor.shape}")
    if history.shape[1] != HISTORY * OLD_OBS:
        raise ValueError(f"Unexpected flat history input: {history.shape}")
    if critic.shape[1] != 214 or checkpoint.get("iter", -1) < 0:
        raise ValueError("Unexpected flat critic or iteration")
    actor_new = actor.new_zeros((actor.shape[0], LATENT + NEW_OBS + COMMANDS))
    actor_new[:, :LATENT + OLD_OBS] = actor[:, :LATENT + OLD_OBS]
    actor_new[:, LATENT + NEW_OBS:] = actor[:, LATENT + OLD_OBS:]
    model["actor.0.weight"] = actor_new

    critic_new = critic.new_zeros((critic.shape[0], critic.shape[1] + NEW_OBS - OLD_OBS))
    critic_new[:, :critic.shape[1] - COMMANDS] = critic[:, :-COMMANDS]
    critic_new[:, -COMMANDS:] = critic[:, -COMMANDS:]
    model["critic.0.weight"] = critic_new

    history_new = history.new_zeros((history.shape[0], HISTORY, NEW_OBS))
    history_new[:, :, :OLD_OBS] = history.reshape(history.shape[0], HISTORY, OLD_OBS)
    encoder["encoder.0.weight"] = history_new.flatten(start_dim=1)
    checkpoint["optimizer_state_dict"] = None
    checkpoint["encoder_optimizer_state_dict"] = None
    checkpoint["iter"] = 0
    checkpoint["task_state"] = None
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve() or args.destination.exists():
        parser.error("Destination must be a new path distinct from the source.")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint = expand_checkpoint(checkpoint)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.destination)
    print(f"Expanded checkpoint: {args.destination}")


if __name__ == "__main__":
    main()
