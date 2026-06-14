# SST Decoding

**EEG decoding of inhibitory control in the stop-signal task (SST).**

This repo contains code used to train models to classify "go" vs "stop" trials from epoched scalp EEG. Included artifacts: cross-subject generalization (leave-one-subject-out), interpretability (especially the stop-related P3).

## How to read this repo

tl;dr: most driver files are in the [`sst_campaign`](sst_campaign) directory, most modular components are in the [`pipeline`](pipeline) directory.

Also, I have mostly written code in strongly typed languages before this and that is why almost all variables have types assigned. Makes python tolerable to me.

I also used "campaign" lingo a lot, because at some point i rewrote a lot of this code to make it more structured, and I couldn't think of a better word. But a campaign is essentially a collection of training runs and some additional scripts.

1. **This README**
2. [`sst_campaign/README.md`](sst_campaign/README.md) - explains the campaign system and how to run the core experiments
3. [`sst_campaign/experiments/README.md`](sst_campaign/experiments/README.md) - lists the scripts that produce the paper results
4. The main model files:
   - [`pipeline/models/cnn_transformer.py`](pipeline/models/cnn_transformer.py)
   - [`pipeline/models/transformer_only.py`](pipeline/models/transformer_only.py)
   - [`pipeline/models/pure_cnn.py`](pipeline/models/pure_cnn.py)
   - [`pipeline/models/enigma_style.py`](pipeline/models/enigma_style.py)
5. [`sst_campaign/run_campaign.py`](sst_campaign/run_campaign.py) - main driver for running everything
6. Analysis/interp scripts:
   - `analyze_attention.py`
   - `interp.py`
   - `explainer.py`

## Reproducing Results

```zsh
uv run python -m sst_campaign.run_campaign \
  --config sst_campaign/configs/full_matrix_parallel.toml \
  --campaign-label paper_full \
  --stop-on-error

uv run python -m sst_campaign.run_manuscript_pipeline \
  --config sst_campaign/configs/manuscript.toml \
  --source-campaign-root sst_campaign_runs/<campaign_root> \
  --output-dir sst_campaign_runs/paper_outputs_final
```

See [`docs/reproducing_paper.md`](docs/reproducing_paper.md) as well.

## venv stuff

I use uv.
