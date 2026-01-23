#!/usr/bin/env python3
import click
from analysis.plots import plot_rmse_by_horizon, plot_rmse_by_cgm_interval, plot_count_by_cgm_interval, plot_rmse_by_demographics


@click.group()
def cli():
    """Analyze and visualize benchmark results."""
    pass


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (e.g., AZT1D, CTR3)')
@click.option('--cgm_range', default=None, help='Filter by CGM range: <70, <50, 70-140, 70-180, >180, >250')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
def horizon_plot(results_dir, dataset, cgm_range, save_path, show):
    """Generate RMSE by forecast horizon plot."""
    # Validate CGM range if provided
    valid_ranges = ["<70", "<50", "70-140", "70-180", ">180", ">250"]
    if cgm_range and cgm_range not in valid_ranges:
        click.echo(f"Error: Invalid CGM range. Must be one of: {', '.join(valid_ranges)}", err=True)
        return
    
    description_parts = ["Generating RMSE by forecast horizon plot"]
    if dataset:
        description_parts.append(f"for dataset: {dataset}")
    if cgm_range:
        description_parts.append(f"filtered by CGM range: {cgm_range} mg/dL")
    if not dataset and not cgm_range:
        description_parts.append("for all datasets combined")
    
    print(" ".join(description_parts) + "...")
    plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset, 
                        cgm_filter=cgm_range, save_path=save_path, show=show)


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
@click.option('--horizons', default='all', help='Forecast horizons: "all" or comma-separated indices (0-11)')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
@click.option('--ylim', default=None, help='Y-axis limits as "min,max" (e.g., "0,1000")')
def count_by_cgm_plot(results_dir, dataset, horizons, save_path, show, ylim):
    """Generate count by CGM interval plot."""
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
    
    # Parse ylim parameter
    ylim_tuple = None
    if ylim:
        try:
            ylim_parts = [float(x.strip()) for x in ylim.split(',')]
            if len(ylim_parts) != 2:
                raise ValueError("ylim must have exactly 2 values")
            ylim_tuple = (ylim_parts[0], ylim_parts[1])
        except ValueError as e:
            click.echo(f"Error parsing ylim: {e}. Expected format: 'min,max'", err=True)
            return
    
    if dataset:
        print(f"Generating count by CGM interval plot for dataset: {dataset}")
    else:
        print("Generating count by CGM interval plot for all datasets combined...")
    
    plot_count_by_cgm_interval(results_dir=results_dir, dataset_filter=dataset, 
                               horizons=horizon_indices, save_path=save_path, show=show, ylim=ylim_tuple)


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--dataset', default=None, help='Filter to specific dataset (omit for all datasets combined)')
@click.option('--demographic', default='age', help='Demographic column name (default: age)')
@click.option('--bin_size', default=10, type=float, help='Interval size for demographic bins (default: 10)')
@click.option('--horizons', default='all', help='Forecast horizons: "all" or comma-separated indices (0-11)')
@click.option('--save_path', default=None, help='Path to save the plot')
@click.option('--show/--no-show', default=True, help='Whether to display the plot')
@click.option('--ylim', default=None, help='Y-axis limits as "min,max" (e.g., "0,80")')
def demographic_plot(results_dir, dataset, demographic, bin_size, horizons, save_path, show, ylim):
    """Generate RMSE by demographic plot."""
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
    
    # Parse ylim parameter
    ylim_tuple = None
    if ylim:
        try:
            ylim_parts = [float(x.strip()) for x in ylim.split(',')]
            if len(ylim_parts) != 2:
                raise ValueError("ylim must have exactly 2 values")
            ylim_tuple = (ylim_parts[0], ylim_parts[1])
        except ValueError as e:
            click.echo(f"Error parsing ylim: {e}. Expected format: 'min,max'", err=True)
            return
    
    if dataset:
        print(f"Generating RMSE by {demographic} plot for dataset: {dataset}")
    else:
        print(f"Generating RMSE by {demographic} plot for all datasets combined...")
    
    plot_rmse_by_demographics(results_dir=results_dir, dataset_filter=dataset,
                             demographic=demographic, bin_size=bin_size,
                             horizons=horizon_indices, save_path=save_path, show=show, ylim=ylim_tuple)


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


@cli.command()
@click.option('--results_dir', default='results', help='Path to results directory')
@click.option('--output_dir', default='figures', help='Directory to save all plots')
@click.option('--individual_datasets', is_flag=True, help='Whether to compute for all individual datasets')
def generate_all_combinations(results_dir, output_dir, individual_datasets):
    """Generate all possible plot combinations for all datasets and CGM ranges."""
    import os
    from analysis.results_loader import get_available_datasets
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available datasets
    datasets_dict = get_available_datasets(results_dir)
    all_datasets = set()
    for model_datasets in datasets_dict.values():
        all_datasets.update(model_datasets)
    all_datasets = sorted(list(all_datasets))
    
    # Define CGM ranges for horizon plots
    cgm_ranges = ["<70", "<50", "70-140", "70-180", ">180", ">250"]
    
    # Define horizon combinations for CGM interval plots
    horizon_combinations = [
        ("all", "all"),
        ("15min", [2]),
        ("30min", [5]), 
        ("45min", [8]),
        ("60min", [11]),
    ]
    
    total_plots = 0
    
    print("Generating all possible plot combinations...")
    print(f"Found datasets: {all_datasets}")
    print(f"Output directory: {output_dir}")
    
    # 1. Horizon plots - All datasets combined
    print("\n=== HORIZON PLOTS (All Datasets) ===")
    
    # Basic horizon plot (no CGM filter)
    save_path = f"{output_dir}/horizon_all_datasets.png"
    print(f"Generating: {save_path}")
    plot_rmse_by_horizon(results_dir=results_dir, save_path=save_path, show=False)
    total_plots += 1
    
    # Horizon plots with CGM filters
    for cgm_range in cgm_ranges:
        safe_range = cgm_range.replace("<", "lt").replace(">", "gt").replace("-", "to")
        save_path = f"{output_dir}/horizon_all_datasets_cgm_{safe_range}.png"
        print(f"Generating: {save_path}")
        plot_rmse_by_horizon(results_dir=results_dir, cgm_filter=cgm_range, 
                           save_path=save_path, show=False)
        total_plots += 1

    print("\n=== CGM INTERVAL PLOTS (All Datasets) ===")
    for horizon_name, horizon_indices in horizon_combinations:
        save_path = f"{output_dir}/cgm_interval_all_datasets_{horizon_name}.png"
        print(f"Generating: {save_path}")
        plot_rmse_by_cgm_interval(results_dir=results_dir, horizons=horizon_indices,
                                  save_path=save_path, show=False)
        total_plots += 1

    if individual_datasets:
        print("\n=== HORIZON PLOTS (Individual Datasets) ===")
        for dataset in all_datasets:
            os.makedirs(f'{output_dir}/{dataset}', exist_ok=True)

            # Basic horizon plot for dataset
            save_path = f"{output_dir}/{dataset}/horizon_{dataset}.png"
            print(f"Generating: {save_path}")
            plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset,
                               save_path=save_path, show=False)
            total_plots += 1

            # Horizon plots with CGM filters for each dataset
            for cgm_range in cgm_ranges:
                safe_range = cgm_range.replace("<", "lt").replace(">", "gt").replace("-", "to")
                save_path = f"{output_dir}/{dataset}/horizon_{dataset}_cgm_{safe_range}.png"
                print(f"Generating: {save_path}")
                plot_rmse_by_horizon(results_dir=results_dir, dataset_filter=dataset,
                                   cgm_filter=cgm_range, save_path=save_path, show=False)
                total_plots += 1

        # 4. CGM Interval plots - Individual datasets
        print("\n=== CGM INTERVAL PLOTS (Individual Datasets) ===")
        for dataset in all_datasets:
            for horizon_name, horizon_indices in horizon_combinations:
                save_path = f"{output_dir}/{dataset}/cgm_interval_{dataset}_{horizon_name}.png"
                print(f"Generating: {save_path}")
                plot_rmse_by_cgm_interval(results_dir=results_dir, dataset_filter=dataset,
                                         horizons=horizon_indices, save_path=save_path, show=False)
                total_plots += 1

    print(f"\n🎉 Generated {total_plots} plots in {output_dir}/")
    print(f"\nPlot categories generated:")
    print(f"  - Horizon plots (all datasets): {1 + len(cgm_ranges)}")
    print(f"  - Horizon plots (per dataset): {len(all_datasets) * (1 + len(cgm_ranges))}")
    print(f"  - CGM interval plots (all datasets): {len(horizon_combinations)}")
    print(f"  - CGM interval plots (per dataset): {len(all_datasets) * len(horizon_combinations)}")


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
        if sys.argv[1] in ['horizon-plot', 'cgm-interval-plot', 'count-by-cgm-plot', 'demographic-plot', 'all-plots', 'generate-all-combinations']:
            cli()
        else:
            # Old style arguments - run main
            main()