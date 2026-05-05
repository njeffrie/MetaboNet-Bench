# Glucose Prediction Scripts

Tools for evaluating glucose-prediction models from `combined_results*.parquet` files.
Each parquet has one row per (model, sample) with columns `label_t0..t11` and
`pred_t0..t11` (12 prediction horizons in 5-minute steps, so t0 = 5 min, t11 = 60 min).

## Pipelines

### `combine_results.py`
** RUN THIS FIRST **
Accepts results files/directory in the form of per-model parquet files or a single 
results file with one model per column.

Aggregates per-model prediction parquets that result from running the models on the 
test set with the original MetaboNet dataset into a single `combined_results*.parquet` 
that every plot script consumes.

## Shared helpers

### `model_styles.py`
Central place for colors, legends, model-filter CLI flags, and render-order rules
used by every plot script. Defines `ABLATION_MODELS` (the GluForecast feature-set
ablation set), the `--ablation`/`--all-variants`/`--exclude-models` flags, the
arch-vs-feature color schemes, and the rule that gluforecast variants always
render on top.

## All-models-on-one-axes plots

These all share the same model-filter flags
(`--exclude-models`, `--all-variants`, `--ablation {all,only,exclude}`),
default to one variant per architecture, color by architecture, and accept
`--figsize W H`. The one-panel-per-model plots below use the same filter
flags.

### `plot_model_rmse.py`
RMSE vs prediction horizon (5–60 min), one line per model. Has a mirrored
mmol/L axis on the right.

### `plot_rmse_by_bg.py`
RMSE vs reference CGM value, one line per model. Always emits a companion
`*_support.png` showing the sample-count distribution per CGM bin, with the
same x-axis. Two layouts: `--combined` (all models on one axes) or default
grid (one panel per model).

### `plot_rmse_by_correction.py`
RMSE across all horizons split into two subplots:
*hyperglycemia (BG > 250) **with** a recent correction bolus* vs *without one*.
Also writes a difference plot.

### `plot_rmse_by_meal_presence.py`
RMSE across all horizons split into two subplots:
*no recent meal* (carbs == 0 across the lookback) vs *recent meal present*.
Also writes a difference plot.

### `plot_rmse_by_recent_carbs.py`
RMSE at the 30-min horizon partitioned by *when* carbs were consumed
(t0, t-5, t-10, …, t-25 min). Emits both a bar version and a line version.

### `plot_rmse_by_variable.py`
RMSE vs a continuous variable (`age`, `cgm_mean`, `cgm_std`, `weight`,
`height`, `age_of_diagnosis`, `years_since_diagnosis`). Bins the data and
shows mean RMSE per bin with standard-error bars; one line per model.
Optional `--show-scatter` overlays raw points.

### `plot_rmse_by_demographic.py`
Grouped bar chart of RMSE by model at one prediction horizon, stratified by a
demographic variable (`age`, `gender`, `ethnicity`, `insulin_delivery_modality`,
`cgm_device`, `subject_split_across_traintest`). `--bars-by` toggles whether
demographics or models are on the x-axis.

## One-panel-per-model plots

These use the shared model-filter flags too, but render a grid with one
panel per model rather than overlaying everything.

### `plot_rmse_by_demographic_horizon.py`
For each model, a line plot of RMSE vs horizon with one line per demographic
group. Use this to see how subgroup error scales with horizon.

### `plot_rmse_2d.py`
Contour plot of patient-level RMSE over a 2D feature space (e.g., age × CGM
mean), one subplot per model. Uses `scipy.griddata` to interpolate sparse
per-patient RMSE points onto a regular grid.

### `plot_rmse_heatmap.py`
Heatmap of RMSE with reference BG bins on the x-axis. Two y-axis modes:
`bg-horizon` (5–60 min) or `bg-cgm-std` (CGM-variability bins for a single
horizon). One heatmap per model; useful for localizing where each model
struggles.

### `plot_rmse_scatter.py`
Scatter of per-patient RMSE vs a continuous variable, with a LOWESS smoother
overlay; one panel per model. Shows the raw distribution and trend without
binning.

### `plot_signed_error_dist.py`
3D ridge plot of the distribution of signed errors (`pred - label`) per model
per horizon, computed via Gaussian KDE. Reveals bias direction and how error
spread evolves with horizon.

## Driver

### `run_all.sh`
Convenience wrapper that runs the above plots end-to-end against a chosen
input parquet.

## Common flags (most plots)

| Flag                          | Meaning                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `--input PATH`                | Input parquet (default: `combined_results_new.parquet`).                |
| `--output PATH`               | Output PNG (auto-generated if omitted).                                 |
| `--exclude-models M [M ...]`  | Drop these models. Default drops `glucose_decoder`, `gluformer-tiny`.   |
| `--all-variants`              | Show every variant. Default keeps the fullest variant per architecture. |
| `--ablation {all,only,exclude}` | `only` = just the GluForecast ablation set (colored by feature). `exclude` = drop them. `all` (default) = keep them. |
| `--figsize W H`               | Figure size in inches. Some scripts also expose `--figsize-diff`, `--figsize-grid`, or `--figsize-support`. |
