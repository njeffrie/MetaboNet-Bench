python scripts/glucose_prediction/plot_rmse_by_bg.py --input combined_results_with_aux.parquet --combined --output rmse_vs_bg_h30.png --horizon 30
python scripts/glucose_prediction/plot_rmse_by_bg.py --input combined_results_with_aux.parquet --combined --output rmse_vs_bg_h60.png --horizon 60
python scripts/glucose_prediction/plot_rmse_by_meal_presence.py --input combined_results_with_aux.parquet --output results_metabonet_bench_meal --exclude-models zoh linear gluformer --figsize 10 4 --figsize-diff 7 4
python scripts/glucose_prediction/plot_rmse_by_correction.py --input combined_results_with_aux.parquet --output results_metabonet_bench_correction --exclude-models zoh linear gluformer --figsize 10 4 --figsize-diff 7 4
python scripts/glucose_prediction/plot_model_rmse.py --input combined_results_with_aux.parquet --ablation exclude
