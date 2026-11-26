#!/usr/bin/env python3
"""Mask all on-diagonal (intrachain) covariance blocks."""

from _run_masking_strategy import run_strategy


if __name__ == "__main__":
    run_strategy("ondiag", __doc__)
