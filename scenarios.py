# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
scenarios.py — backward-compatibility shim.

All scenario and grader definitions have been moved to tasks.py.
This module re-exports them so existing imports continue to work.
"""

from tasks import SCENARIOS, GRADERS, TASKS, MIN_SCORE, MAX_SCORE  # noqa: F401

__all__ = ["SCENARIOS", "GRADERS", "TASKS", "MIN_SCORE", "MAX_SCORE"]