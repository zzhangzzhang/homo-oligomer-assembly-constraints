#!/usr/bin/env python3
"""Mask all off-diagonal (inter-chain) covariance blocks."""

from _run_masking_strategy import run_strategy


if __name__ == "__main__":
    run_strategy("offdiag", __doc__)
