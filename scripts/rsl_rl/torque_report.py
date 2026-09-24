"""Simulator-independent torque averaging and per-episode export (all units N m)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCES = ('pd_estimate', 'solver_joint_effort')
SOURCE_DESCRIPTIONS = {
    'pd_estimate': 'Implicit PD pre-step estimate, clipped to actuator effort limits; NOT measured motor torque.',
    'solver_joint_effort': 'PhysX incoming joint force projected onto the DOF motion axis, after the physics step; '
                           'may include constraints/contact response; NOT isolated motor torque.',
}


def block_mean(values, window):
    """Average consecutive physics samples, keeping any short final block."""
    return np.stack([values[i:i + window].mean(axis=0) for i in range(0, len(values), window)])


def statistics(values, names):
    return {name: {'mean_Nm': float(values[:, i].mean()),
                   'mean_abs_Nm': float(np.abs(values[:, i]).mean()),
                   'rms_Nm': float(np.sqrt(np.square(values[:, i]).mean())),
                   'min_Nm': float(values[:, i].min()),
                   'max_Nm': float(values[:, i].max()),
                   'peak_abs_Nm': float(np.abs(values[:, i]).max())}
            for i, name in enumerate(names)}


def aligned_mean(episodes):
    """Align by elapsed recovery time; never zero-pad or repeat ended episodes."""
    totals = np.zeros((max(map(len, episodes)), *episodes[0].shape[1:]))
    counts = np.zeros(len(totals), dtype=int)
    for episode in episodes:
        totals[:len(episode)] += episode
        counts[:len(episode)] += 1
    return totals / counts.reshape((-1,) + (1,) * (totals.ndim - 1)), counts


def write_curve(path, times, values, names, counts=None):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_end_s', *(['contributing_episodes'] if counts is not None else []),
                         *[f'{src}/{name}_Nm' for src in SOURCES for name in names]])
        for i, time in enumerate(times):
            writer.writerow([time, *([int(counts[i])] if counts is not None else []), *values[i].reshape(-1)])


def plot_curve(path, times, values, names, title):
    order = [names.index(f'{kind}_{side}_Joint') for kind in ('abad', 'hip', 'knee', 'wheel')
             for side in ('L', 'R') if f'{kind}_{side}_Joint' in names]
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for ax, joint in zip(axes.flat, order):
        ax.plot(times, values[:, 0, joint], lw=1, label='Clipped PD estimate')
        ax.plot(times, values[:, 1, joint], lw=.8, alpha=.75, label='Solver joint-axis effort')
        ax.set_title(names[joint])
        ax.set_ylabel('Torque (N m)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time since recovery reset (s)')
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


class TorqueReport:
    def __init__(self, directory, names, physics_dt, decimation, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.names = names
        self.dt = physics_dt
        self.decimation = decimation
        self.metadata = dict(metadata, torque_sources=SOURCE_DESCRIPTIONS, joint_names=names,
                             physics_dt_s=physics_dt, control_dt_s=physics_dt * decimation,
                             time_definition='End of sampled physics/control interval. PD estimate belongs to its start.',
                             average_definition='Signed arithmetic mean of physics samples per control interval. '
                             'Peaks and RMS use unaveraged physics samples.')
        (self.directory / 'metadata.json').write_text(json.dumps(self.metadata, indent=2))
        self.episodes = []
        self.curves = []

    def save_episode(self, samples, info):
        values = np.asarray(samples, dtype=float)
        if values.ndim != 3 or values.shape[1:] != (2, len(self.names)) or not np.isfinite(values).all():
            raise ValueError('Invalid torque samples')
        episode_id = len(self.episodes)
        dest = self.directory / f'episode_{episode_id:04d}'
        dest.mkdir()
        mean = block_mean(values, self.decimation)
        times = np.minimum(np.arange(1, len(mean) + 1) * self.decimation, len(values)) * self.dt
        write_curve(dest / 'physics_torques.csv', np.arange(1, len(values) + 1) * self.dt, values, self.names)
        write_curve(dest / 'mean_torques.csv', times, mean, self.names)
        result = dict(info, episode_id=episode_id, samples=len(values), duration_s=len(values) * self.dt,
                      joints={src: statistics(values[:, s], self.names) for s, src in enumerate(SOURCES)})
        (dest / 'summary.json').write_text(json.dumps(result, indent=2))
        plot_curve(dest / 'mean_torques.png', times, mean, self.names,
                   f'Episode {episode_id} | {info["outcome"]} | Mean per control interval')
        self.episodes.append(result)
        if info['complete']:
            self.curves.append(mean)
        self.save_summary()

    def save_summary(self):
        # Persist after every episode, so completed episodes survive interruption.
        (self.directory / 'summary.json').write_text(json.dumps(
            dict(self.metadata, episodes=self.episodes), indent=2))
        with (self.directory / 'maxima.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['episode_id', 'complete', 'outcome', 'source', 'joint',
                             'mean_Nm', 'mean_abs_Nm', 'rms_Nm', 'min_Nm', 'max_Nm', 'peak_abs_Nm'])
            for episode in self.episodes:
                for src in SOURCES:
                    for joint in self.names:
                        row = episode['joints'][src][joint]
                        writer.writerow([episode['episode_id'], episode['complete'], episode['outcome'], src, joint,
                                         *[row[k] for k in ('mean_Nm', 'mean_abs_Nm', 'rms_Nm', 'min_Nm', 'max_Nm', 'peak_abs_Nm')]])

    def finish(self):
        if self.episodes:
            with (self.directory / 'overall_maxima.csv').open('x', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['source', 'joint', 'min_Nm', 'max_Nm', 'peak_abs_Nm', 'peak_episode_id'])
                for src in SOURCES:
                    for joint in self.names:
                        peak = max(self.episodes, key=lambda ep: ep['joints'][src][joint]['peak_abs_Nm'])
                        writer.writerow([src, joint,
                            min(ep['joints'][src][joint]['min_Nm'] for ep in self.episodes),
                            max(ep['joints'][src][joint]['max_Nm'] for ep in self.episodes),
                            peak['joints'][src][joint]['peak_abs_Nm'], peak['episode_id']])
        if self.curves:
            mean, counts = aligned_mean(self.curves)
            times = np.arange(1, len(mean) + 1) * self.dt * self.decimation
            write_curve(self.directory / 'episodes_mean.csv', times, mean, self.names, counts)
            plot_curve(self.directory / 'episodes_mean.png', times, mean, self.names,
                       f'{len(self.curves)} complete episodes | Time-aligned signed mean (see CSV for sample counts)')
