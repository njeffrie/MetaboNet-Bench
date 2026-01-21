#!/usr/bin/env python3
import click
from analysis.plots import plot_rmse_by_horizon, plot_rmse_by_cgm_interval


@click.group()
def cli():
    """Analyze and visualize benchmark results."""
    pass


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
def horizon_plot(results_dir, dataset, save_path, show):
    """Generate RMSE by forecast horizon plot."""
    if dataset:
        print(f"Generating RMSE by forecast horizon plot for dataset: {dataset}")
    else:
        print("Generating RMSE by forecast horizon plot for all datasets combined...")
    plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset, save_path=save_path, show=show)


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--horizons', default='all', help='Forecast horizons: "all" or comma-separated indices (0-11)')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
def cgm_interval_plot(results_dir, dataset, horizons, save_path, show):
    """Generate RMSE by CGM interval plot."""
    # Parse horizons parameter
    if horizons == 'all':
        horizon_indices = 'all'
    else:
        try:
            horizon_indices = [int(h.strip()) for h in horizons.split(',')]
            # Validate horizon indices
            if not all(0 <= h <= 11 for h in horizon_indices):
                raise ValueError("Horizon indices must be between 0 and 11")
        except ValueError as e:
            click.echo(f"Error parsing horizons: {e}", err=True)
            return
    
    if dataset:
        print(f"Generating RMSE by CGM interval plot for dataset: {dataset}")
    else:
        print("Generating RMSE by CGM interval plot for all datasets combined...")
    
    plot_rmse_by_cgm_interval(results_dir=results_dir, dataset_filter=dataset, 
                              horizons=horizon_indices, save_path=save_path, show=show)


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--output_dir', default='plots', help='Directory to save all plots')
def all_plots(results_dir, dataset, output_dir):
    """Generate all analysis plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = f"_{dataset}" if dataset else "_all_datasets"
    
    print("Generating all analysis plots...")
    
    # RMSE by horizon plot
    horizon_save_path = f"{output_dir}/rmse_by_horizon{suffix}.png"
    print(f"1. Generating RMSE by forecast horizon plot...")
    plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset, 
                        save_path=horizon_save_path, show=False)
    
    # RMSE by CGM interval plot (all horizons)
    interval_save_path = f"{output_dir}/rmse_by_cgm_interval{suffix}.png"
    print(f"2. Generating RMSE by CGM interval plot...")
    plot_rmse_by_cgm_interval(results_dir=results_dir, dataset_filter=dataset, 
                             horizons='all', save_path=interval_save_path, show=False)
    
    print(f"\nAll plots saved to {output_dir}/")


# Keep backwards compatibility - default command when no subcommand given
@click.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--output_dir', default='plots', help='Directory to save all plots')
def main(results_dir, dataset, output_dir):
    """Generate all analysis plots (default behavior)."""
    all_plots.callback(results_dir, dataset, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments - run main
        main()
    else:
        # Check if first argument is a subcommand
        if sys.argv[1] in ['horizon-plot', 'cgm-interval-plot', 'all-plots']:
            cli()
        else:
            # Old style arguments - run main
            main()