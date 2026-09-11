#!/usr/bin/env python
"""Evaluate a trained PAW-augmentation MACE checkpoint on a test split.

Same CLI as train.py but forces --eval-only: no training, the test split is
replayed and the usual test_metrics_* files plus per-structure predictions
are written. Point --resume (or the config) at the checkpoint to score.
"""
import sys

from neural_init.augnet.train_augnet import main

if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        sys.argv.append("--eval-only")
    main()
