# `sst_campaign/experiments/`

## Scripts

### Standard baseline / grouped-CV training
- `train_repro_baseline.py` - standard training, test-train split, multisubject

### Strict held-out subject (zero-shot LOSO)
- `train_zero_shot_loso.py` - leaves a single subject completely out of the training set. All testing is done on this subject. This is repeated for each subject (so # subject-fold cross-validation)

## Example

```bash
uv run python -m sst_campaign.experiments.train_repro_baseline \
  --subjects 1 2 \
  --model cnn_transformer \
  --cv-mode loso \
  --epochs 1 \
  --patience 1 \
  --batch-size 16 \
  --device cpu \
  --output-dir sst_campaign_runs/test \
  --run-label test
```
