# `sst_campaign/`

Campaign code for the SST decoding repo.

Contains:

- TOML configs and the campaign runner
- direct experiment entrypoints used by the campaign runner
- generic post-hoc interpretability / explainer / attention analyses
- optional follow-up analyses for time generalization, channel ablation, and causal go/stop ablation


## Example

```bash
uv run python -m sst_campaign.run_campaign \
  --config sst_campaign/configs/smoke.toml \
  --campaign-label test \
  --stop-on-error
```


## Configs

| Config                                         | Device default | Subjects                | Models                         | Minimal-try           | Intended use                           |
| ---------------------------------------------- | -------------- | ----------------------- | ------------------------------ | --------------------- | -------------------------------------- |
| `configs/smoke.toml`                           | CPU            | S1, S2                  | all 3 base families    | S1 only, tiny grid    | Fast end-to-end validation             |
| `configs/full_matrix_parallel.toml`            | CUDA           | all discovered subjects | the 5 core families    | disabled              | Opt-in weighted parallel full campaign |


## Output

Each invocation creates a timestamped campaign root under `campaign.output_root` and writes:

- `campaign_config_snapshot.json` - frozen input config.
- `logs/task_*.stdout.log` and `logs/task_*.stderr.log` - child process logs.
- `artifacts/` - versioned task output directories.
- `campaign_metadata.json` - task metadata.
- `campaign_run_index.csv` - flat task/run table.
- `campaign_failures.csv` - failed task summaries.
- `aggregate_metrics.csv` - metrics copied from completed task metadata.
- `paper_summary.md` - compact human-readable summary.

## Extra analysis stuff on completed runs

- `run_causal_go_stop_cycle.py`
