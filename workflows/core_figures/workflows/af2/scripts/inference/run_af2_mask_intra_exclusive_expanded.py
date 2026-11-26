#!/usr/bin/env python3
"""Mask off-diagonal blocks and expanded intra-exclusive diagonal coordinates."""

from _run_masking_strategy import run_strategy


if __name__ == "__main__":
    run_strategy("intra_expand", __doc__)
