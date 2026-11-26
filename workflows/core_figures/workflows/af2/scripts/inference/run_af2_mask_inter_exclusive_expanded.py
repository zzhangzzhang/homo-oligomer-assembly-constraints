#!/usr/bin/env python3
"""Mask off-diagonal blocks and expanded inter-exclusive diagonal coordinates."""

from _run_masking_strategy import run_strategy


if __name__ == "__main__":
    run_strategy("inter_expand", __doc__)
