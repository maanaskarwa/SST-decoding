"""CLI wrapper that runs the baseline pipeline in leave-one-subject-out mode for zero-shot cross-subject evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.train.driver import set_random_seed

from pipeline.run_versioning import prepare_versioned_output_dir
from sst_campaign.experiments.train_repro_baseline import train_run
from sst_campaign.utils.common import emit_run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dedicated leave-one-subject-out (zero-shot participant) training/evaluation "
            "with reproducible saved checkpoints and splits."
        )
    )
    from sst_campaign.experiments.common import add_common_training_args

    add_common_training_args(parser, loso=True)
    # loso wrapper forces cv_mode later; output dir default is different from baseline
    parser.set_defaults(output_dir="decoding_results_zero_shot_loso")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cv_mode = "loso"

    root = Path(args.root).resolve()
    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = root / out_base
    out_dir, version_meta = prepare_versioned_output_dir(
        base_output_dir=out_base,
        experiment_name="train_zero_shot_loso",
        config=vars(args),
        disable_versioning=bool(args.disable_versioning),
    )
    print(f"Run output dir: {out_dir}")
    emit_run_output_dir(out_dir)
    print(f"Run version meta: {json.dumps(version_meta)}")

    set_random_seed(int(args.random_state))
    train_run(args=args, root=root, out_dir=out_dir)


if __name__ == "__main__":
    main()
