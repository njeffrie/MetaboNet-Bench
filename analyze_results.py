#!/usr/bin/env python3
import click
from analysis.plots import plot_rmse_by_horizon


@click.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
def main(results_dir, dataset, save_path, show):
    """Analyze and visualize benchmark results."""
    if dataset:
        print(f"Generating RMSE by forecast horizon plot for dataset: {dataset}")
    else:
        print("Generating RMSE by forecast horizon plot for all datasets combined...")
    plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset, save_path=save_path, show=show)


if __name__ == "__main__":
    main()