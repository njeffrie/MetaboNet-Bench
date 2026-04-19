"""
Rough wall-clock estimate for Optuna (two studies) + ablation training on one GPU.

Plug in sec/epoch from bench.py (or your own timings). Uses average epochs per
trial when early stopping is enabled.

Example:
    python -m studies.feature_ablation.estimate_runtime \\
        --lstm-sec-per-epoch 2.5 --units-sec-per-epoch 40 \\
        --n-trials 30 --max-epochs-per-trial 20 --avg-epochs-per-trial 12 \\
        --ablation-runs 8 --ablation-epochs 30 --margin-frac 0.15
"""

from __future__ import annotations

import json

import click

@click.command()
@click.option('--lstm-sec-per-epoch', type=float, default=None,
              help='Mean train+val seconds per epoch for LSTM (not needed with --bench-json)')
@click.option('--units-sec-per-epoch', type=float, default=None,
              help='Mean train+val seconds per epoch for UniTS')
@click.option('--n-trials', type=int, default=30, help='Trials per model study')
@click.option('--max-epochs-per-trial', type=int, default=20,
              help='Cap passed to Optuna training')
@click.option('--avg-epochs-per-trial', type=float, default=None,
              help='Expected epochs per trial after early stopping (default: 0.65 * max)')
@click.option('--ablation-runs', type=int, default=8,
              help='Total final train jobs (default 8 = 4 LSTM + 4 UniTS)')
@click.option('--lstm-ablation-runs', type=int, default=None,
              help='Override: how many LSTM ablation jobs (default ablation_runs//2)')
@click.option('--units-ablation-runs', type=int, default=None,
              help='Override: how many UniTS ablation jobs (default ablation_runs//2)')
@click.option('--ablation-epochs', type=int, default=50,
              help='--epochs for each ablation job')
@click.option('--avg-epochs-ablation', type=float, default=None,
              help='If set, expected epochs per ablation run (else ablation-epochs)')
@click.option('--margin-frac', type=float, default=0.15,
              help='Safety margin on total time (checkpoint I/O, variance)')
@click.option('--bench-json', type=str, default=None,
              help='Optional JSON from bench.py (reads lstm/units sec_per_epoch)')
def main(
    lstm_sec_per_epoch: float | None,
    units_sec_per_epoch: float | None,
    n_trials: int,
    max_epochs_per_trial: int,
    avg_epochs_per_trial: float | None,
    ablation_runs: int,
    lstm_ablation_runs: int | None,
    units_ablation_runs: int | None,
    ablation_epochs: int,
    avg_epochs_ablation: float | None,
    margin_frac: float,
    bench_json: str | None,
):
    if bench_json:
        with open(bench_json) as f:
            b = json.load(f)
        lstm_sec_per_epoch = float(b['lstm']['sec_per_epoch'])
        units_sec_per_epoch = float(b['units']['sec_per_epoch'])
        print(f'Loaded bench JSON: lstm {lstm_sec_per_epoch:.4f}s/ep, '
              f'UniTS {units_sec_per_epoch:.4f}s/ep')
    elif lstm_sec_per_epoch is None or units_sec_per_epoch is None:
        raise click.UsageError(
            'Provide --lstm-sec-per-epoch and --units-sec-per-epoch, or --bench-json'
        )

    avg_ep_trial = avg_epochs_per_trial
    if avg_ep_trial is None:
        avg_ep_trial = max(1.0, 0.65 * float(max_epochs_per_trial))

    avg_ep_abl = avg_epochs_ablation
    if avg_ep_abl is None:
        avg_ep_abl = float(ablation_epochs)

    t_lstm_study = n_trials * avg_ep_trial * lstm_sec_per_epoch
    t_units_study = n_trials * avg_ep_trial * units_sec_per_epoch
    t_optuna = t_lstm_study + t_units_study

    n_lstm_ab = lstm_ablation_runs if lstm_ablation_runs is not None else ablation_runs // 2
    n_u_ab = units_ablation_runs if units_ablation_runs is not None else ablation_runs // 2
    t_ablation = (
        n_lstm_ab * avg_ep_abl * lstm_sec_per_epoch
        + n_u_ab * avg_ep_abl * units_sec_per_epoch
    )

    raw_total = t_optuna + t_ablation
    total = raw_total * (1.0 + margin_frac)

    def _fmt(sec: float) -> str:
        h = sec / 3600.0
        return f'{sec / 60.0:.1f} min ({h:.2f} h)'

    print('\n--- Estimate (single GPU, sequential Optuna trials) ---')
    print(f'Optuna LSTM study:  {_fmt(t_lstm_study)}')
    print(f'Optuna UniTS study: {_fmt(t_units_study)}')
    print(f'Optuna total:       {_fmt(t_optuna)}')
    print(
        f'Ablation ({n_lstm_ab} LSTM + {n_u_ab} UniTS, ~{avg_ep_abl:.1f} ep/run): {_fmt(t_ablation)}'
    )
    print(f'Raw sum:            {_fmt(raw_total)}')
    print(f'With margin {margin_frac:.0%}: {_fmt(total)}  ({total/3600.0:.2f} h)\n')

    if total > 24 * 3600:
        print('WARNING: estimate exceeds 24h. Reduce n_trials, max_epochs_per_trial, '
              'or ablation --epochs / number of runs.')
    else:
        headroom_h = (24 * 3600 - total) / 3600.0
        print(f'Headroom vs 24h: ~{headroom_h:.2f} h')


if __name__ == '__main__':
    main()
