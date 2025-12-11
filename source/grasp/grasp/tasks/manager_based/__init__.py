# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based task environments."""

import gymnasium as gym  # noqa: F401

# Import sub-packages to register environments
from . import locomotion_spot  # noqa: F401
from . import locomanipulation_spot  # noqa: F401
