# Reproducing paper outputs

Steps to reproduce the paper's results:

## 1. Run the campaign

```zsh
uv run python -m sst_campaign.run_campaign \
  --config sst_campaign/configs/full_matrix_parallel.toml \
  --campaign-label paper_full \
  --stop-on-error
```

This writes a timestamped campaign root such as:

```text
sst_campaign_runs/paper_full__YYYY-MM-DD_HH-MM-SS
```

Wait until the campaign prints `CAMPAIGN_STATUS status=completed` and writes `campaign_metadata.json`.

Campaign outputs include:

- baseline and LOSO training for the configured model families;
- generic interpretability outputs;
- explainer-suite outputs;
- attention-rollout outputs;
- `campaign_metadata.json`, `aggregate_metrics.csv`, and `paper_summary.md`.

## 2. Rebuild manuscript outputs

Replace `<campaign_root>` with the completed directory from step 1:

```zsh
uv run python -m sst_campaign.run_manuscript_pipeline \
  --config sst_campaign/configs/manuscript.toml \
  --source-campaign-root sst_campaign_runs/<campaign_root> \
  --output-dir sst_campaign_runs/paper_outputs_final
```

If `--output-dir` is omitted, the script writes to a timestamped `sst_campaign_runs/manuscript_rebuild__...` directory. `--source-campaign-root` is required unless a config explicitly sets `manuscript.source_campaign_root`.

Manuscript outputs include:

- `performance/table1_model_performance.csv`;
- `performance/model_comparison_stats.csv`;
- `performance/figure1_precision_recall_loso.{png,pdf,svg}`;
- `performance/figure1b_loso_balanced_accuracy_bar.{png,pdf,svg}`;
- `behavior/behavioral_subject_metrics.csv`;
- `behavior/figure_behavior_ssrt_distribution.{png,pdf,svg}`;
- `behavior/figure_behavior_inhibition_function.{png,pdf,svg}`;
- `behavior/model_behavior_correlations.csv`;
- `interpretability_validation/p3_roi_ablation/` with the P3/ROI perturbation tables;
- `manuscript_pipeline_summary.json`.

## Model order

The model families included in manuscript tables are controlled by `manuscript.model_order` in `sst_campaign/configs/manuscript.toml`. Override it at runtime if needed:

```zsh
uv run python -m sst_campaign.run_manuscript_pipeline \
  --config sst_campaign/configs/manuscript.toml \
  --source-campaign-root sst_campaign_runs/<campaign_root> \
  --output-dir sst_campaign_runs/paper_outputs_final \
  --model-order cnn_transformer transformer_only pure_cnn enigma
```

## Optional extended causal validation

The manuscript rebuild already runs the P3/ROI perturbation outputs needed for the current paper tables. If you also want the longer causal cycle with constrained early-vs-late retraining, run it separately after step 1:

```zsh
uv run python -m sst_campaign.run_causal_go_stop_cycle \
  --source-campaign-root sst_campaign_runs/<campaign_root> \
  --device cuda \
  --output-dir sst_campaign_runs/paper_causal_cycle \
  --run-label paper_causal_cycle
```

## Skip heavy perturbation checks

For a quick manuscript rebuild without P3/ROI ablation:

```zsh
uv run python -m sst_campaign.run_manuscript_pipeline \
  --config sst_campaign/configs/manuscript.toml \
  --source-campaign-root sst_campaign_runs/<campaign_root> \
  --output-dir sst_campaign_runs/paper_outputs_quick \
  --skip-heavy
```
