#!/usr/bin/env python
"""Train a PAW-augmentation MACE model.

Thin entry point over src/train_augnet.py; run `python train.py --help`
for the full CLI. Evaluation of a trained checkpoint goes through eval.py.
"""
from src.train_augnet import main

if __name__ == "__main__":
    main()
