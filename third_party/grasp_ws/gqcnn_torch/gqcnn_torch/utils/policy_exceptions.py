# -*- coding: utf-8 -*-
"""
Policy exceptions for gqcnn_torch package.
"""


class NoValidGraspsException(Exception):
    """Exception raised when no valid grasps are found."""
    pass


class NoAntipodalPairsFoundException(Exception):
    """Exception raised when no antipodal pairs are found during sampling."""
    pass
