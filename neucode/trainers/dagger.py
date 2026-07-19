"""
DAgger (Dataset Aggregation) trainer for NeuCoDe controllers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from neucode.harness.base import BaseHarness


class DAggerTrainer:
    """
    Wraps a replacement trainer and adds the iterative DAgger loop.

    :param harness: Configured harness (sim or hw) whose controller has ``record_state=True``.
    :param expert: Expert for labeling - any object with ``step(sp, y, dt)`` or
        ``predict(sp, y, e)``.  Typically ``PIDState`` (C-core) or a ``Controller``.
    :param wrapped_trainer: SNNReplacementTrainer or ANNReplacementTrainer.
    :param actuator_limits: Dict with ``u_min`` and ``u_max``.
    :param dt: Control loop timestep in seconds.
    :param clip_overrides: Per-feature clip bounds ``{feature_idx: (lo, hi)}``.
    """

    def __init__(
        self,
        harness: BaseHarness,
        expert,
        wrapped_trainer,
        actuator_limits: dict,
        dt: float = 0.01,
        clip_overrides: dict | None = None,
    ):
        self.harness = harness
        self.expert = expert
        self.wrapped_trainer = wrapped_trainer
        self.actuator_limits = actuator_limits
        self.dt = dt
        self.label_dt = None
        self.step_delay = 0.0
        self.clip_overrides = clip_overrides


    def train(
        self,
        setpoints: list[float],
        initial_model_path: str | Path = None,
        initial_dataset_path: str | Path = None,
        model_save_path: str | Path = None,
        scaler_save_path: str | Path = None,
        stats_save_path: str | Path = None,
        n_rounds: int = 3,
        episode_time: float = 8.0,
        epochs: int = 50,
        target_scale: float = 2.0,
        architecture: str = "hybrid",
        population_size: int = 64,
        seed: int = 42,
        max_seed_samples: int | None = None,
        beta: float = 0.92,
        window_size: int = 100,
        experiment=None,
        tag: str = None,
        seed_tag: str = None,
        seed_dataset_tag: str = None,
    ) -> None:
        """Run the full DAgger training loop.

        :param setpoints: Step values used per episode in each round.
        :param initial_model_path: Pre-trained model.pth (seed policy).
        :param initial_dataset_path: Initial BC dataset CSV.
        :param model_save_path: Output path for DAgger-refined model.pth.
        :param scaler_save_path: Output path for scaler.npz.
        :param stats_save_path: Output path for stats.json.
        :param max_seed_samples: Subsample seed dataset to this many rows
            so that HW rollout data has meaningful weight.
        :param experiment: Optional Experiment instance for path resolution.
        :param tag: Output training tag (required when experiment is passed).
        :param seed_tag: Tag of the seed model within the experiment.
        :param seed_dataset_tag: Tag of the seed dataset (defaults to seed_tag).
        """
        # Resolve paths from experiment
        if experiment is not None:
            if tag is None:
                raise ValueError("tag is required when passing experiment")
            model_save_path = model_save_path or experiment.get_model_path(tag)
            scaler_save_path = scaler_save_path or experiment.get_scaler_path(tag)
            stats_save_path = stats_save_path or experiment.get_stats_path(tag)
            if seed_tag:
                initial_model_path = initial_model_path or experiment.get_model_path(seed_tag)
                initial_dataset_path = initial_dataset_path or experiment.get_dataset_path(seed_dataset_tag or seed_tag)

        model_save_path      = Path(model_save_path)
        initial_model_path   = Path(initial_model_path)
        initial_dataset_path = Path(initial_dataset_path)
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        current_model_path = initial_model_path
        agg_rows = self._load_csv_rows(initial_dataset_path)
        print(f"\n[DAgger] Round 0 - BC seed: {len(agg_rows)} samples "
              f"from {initial_dataset_path.name}")

        if max_seed_samples is not None and len(agg_rows) > max_seed_samples:
            import random
            rng = random.Random(seed)

            episodes = {}
            for row in agg_rows:
                eid = row.get('episode_id', '0')
                episodes.setdefault(eid, []).append(row)

            ep_ids = list(episodes.keys())
            rng.shuffle(ep_ids)

            sampled = []
            n_episodes = 0
            for eid in ep_ids:
                if len(sampled) >= max_seed_samples:
                    break
                sampled.extend(episodes[eid])
                n_episodes += 1
            agg_rows = sampled
            print(f"  Subsampled seed to {len(agg_rows)} samples "
                  f"({n_episodes} episodes from {initial_dataset_path.name})")

        max_seed_eid = max((int(r.get('episode_id', 0)) for r in agg_rows), default=-1)
        episode_counter = max_seed_eid + 1

        for round_idx in range(1, n_rounds + 1):
            print(f"\n[DAgger] Round {round_idx}/{n_rounds} - rolling out policy...")

            # Reload weights once per round (on HW this also exports + prompts flash).
            # Derive scaler from the model directory - on round 1 the output
            # scaler doesn't exist yet, but the seed model's scaler does.
            round_scaler = Path(current_model_path).parent / "scaler.npz"
            self.harness.load_weights(
                current_model_path,
                scaler_path=str(round_scaler) if round_scaler.exists() else None,
            )

            new_rows = []
            for sp_val in setpoints:
                states = self.harness.rollout(
                    setpoint_value=sp_val,
                    episode_time=episode_time,
                    dt=self.dt,
                    step_delay=self.step_delay,
                )

                episode_rows = self._label_with_expert(
                    states, episode_id=episode_counter,
                )
                episode_counter += 1
                new_rows.extend(episode_rows)
                print(f"  sp={sp_val:+.1f}  collected {len(episode_rows)} samples")

            agg_rows = agg_rows + new_rows
            print(f"  Aggregated dataset: {len(agg_rows)} samples total")

            agg_csv = model_save_path.parent / f"_dagger_agg_round{round_idx}.csv"
            self._write_csv(agg_rows, agg_csv)

            print(f"  Retraining ({epochs} epochs)...")
            import inspect
            sig = inspect.signature(self.wrapped_trainer.train)
            all_kwargs = dict(
                dataset_path=str(agg_csv),
                model_save_filename=str(model_save_path),
                scaler_save_filename=str(scaler_save_path),
                stats_save_filename=str(stats_save_path),
                epochs=epochs,
                target_scale=target_scale,
                architecture=architecture,
                population_size=population_size,
                seed=seed,
                cache_prefix=str(agg_csv.with_suffix("")),
                initial_model_path=str(current_model_path),
                clip_overrides=self.clip_overrides,
                beta=beta,
                window_size=window_size,
            )
            valid_kwargs = {k: v for k, v in all_kwargs.items()
                           if k in sig.parameters}
            self.wrapped_trainer.train(**valid_kwargs)
            current_model_path = model_save_path
            print(f"  Model saved -> {model_save_path}")

            self._plot_feature_diagnostics(
                agg_csv, scaler_save_path, round_idx, n_rounds)

        print(f"\n[DAgger] Done. Final model: {model_save_path}")


    def _label_with_expert(self, states: list[dict],
                           episode_id: int = 0) -> list[dict]:
        """Label visited states with expert actions.

        .. note:: Coupled to the 5-feature set (sp, y, e, integral, derivative).
           For trainers with a different feature set, make this a callback.
        """
        if not states:
            return []

        self.expert.reset()

        _use_step = hasattr(self.expert, 'step') and not hasattr(self.expert, 'predict')

        rows = []
        for state in states:
            sp   = state['setpoint']
            y    = state['measurement']
            e    = state['error']

            if _use_step:
                u_expert = self.expert.step(sp, float(y), self.label_dt or self.dt)
            else:
                u_expert = self.expert.predict(sp, y, e)

            rows.append({
                'setpoint':         sp,
                'measurement':      y,
                'error':            e,
                'integral_error':   state['integral_error'],
                'derivative_error': state['derivative_error'],
                'control_effort':   u_expert,
                'episode_id':       episode_id,
            })
        return rows

    @staticmethod
    def _plot_feature_diagnostics(agg_csv: Path, scaler_path: Path,
                                  round_idx: int, n_rounds: int) -> None:
        """
        Save a per-round diagnostic plot: feature histograms vs clip/scaler bounds.
        """
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            return

        agg_csv = Path(agg_csv)
        scaler_path = Path(scaler_path)
        if not agg_csv.exists() or not scaler_path.exists():
            return

        df = pd.read_csv(agg_csv)
        with np.load(str(scaler_path)) as s:
            data_min = s['data_min']
            data_scale = s['data_scale']
            clip_min = s['clip_min'] if 'clip_min' in s.files else None
            clip_max = s['clip_max'] if 'clip_max' in s.files else None

        columns = ['setpoint', 'measurement', 'error',
                   'integral_error', 'derivative_error']
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.ravel()

        for i, col in enumerate(columns):
            ax = axes[i]
            vals = df[col].astype(float).values
            ax.hist(vals, bins=80, alpha=0.7, color='steelblue', edgecolor='none')

            lo, hi = data_min[i], data_min[i] + data_scale[i]
            ax.axvline(lo, color='green', ls='--', lw=1, label='scaler')
            ax.axvline(hi, color='green', ls='--', lw=1)

            if clip_min is not None:
                ax.axvline(clip_min[i], color='red', ls='-', lw=1.2, label='clip')
                ax.axvline(clip_max[i], color='red', ls='-', lw=1.2)
                n_at_lo = (vals <= clip_min[i]).sum()
                n_at_hi = (vals >= clip_max[i]).sum()
                pct = 100.0 * (n_at_lo + n_at_hi) / max(len(vals), 1)
                if pct > 0.01:
                    ax.set_xlabel(f'{pct:.1f}% at clip', fontsize=8, color='red')

            ax.set_title(col, fontsize=9)
            if i == 0:
                ax.legend(fontsize=7)

        axes[5].axis('off')
        fig.suptitle(f'DAgger round {round_idx}/{n_rounds}  -  {len(df)} samples',
                     fontsize=11)
        fig.tight_layout()
        out_path = agg_csv.with_name(f'_dagger_features_round{round_idx}.pdf')
        fig.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"  Feature diagnostics -> {out_path.name}")

    @staticmethod
    def _load_csv_rows(path: Path) -> list[dict]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write_csv(rows: list[dict], path: Path) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
