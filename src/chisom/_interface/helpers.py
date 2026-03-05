#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:02:48 2022

@author: j.kaminski@uni-muenster.de for agkoch
"""

import numpy as np
from numba import jit, prange
from pyqtgraph import ColorMap


def create_stops(n_classes: int):
    """
    Create the stops necessary to create a non-continous colormap with n_classes

    Parameters
    ----------
    n_classes : int
        number of discrete colors

    Returns
    -------
    ndarray[np.int32]
        Stops for the color values
    """

    color_start = np.linspace(0, 1, n_classes + 1)
    color_end = np.linspace(0, 1, n_classes + 1) - 0.00001
    stops = np.empty((n_classes * 2), dtype=np.float32)

    for i in range(len(color_start) - 1):
        stops[(i * 2)] = color_start[i]
        stops[(i * 2) + 1] = color_end[i + 1]
    stops[-1] = color_start[-1]

    return stops


EarthColorMap = ColorMap(
    pos=create_stops(13),
    color=[
        (0, 0, 245),
        (0, 0, 245),
        (55, 125, 34),
        (55, 125, 34),
        (62, 131, 38),
        (22, 131, 38),
        (93, 144, 54),
        (93, 144, 54),
        (142, 160, 78),
        (142, 160, 78),
        (189, 176, 101),
        (189, 176, 101),
        (188, 165, 95),
        (188, 165, 95),
        (173, 149, 82),
        (173, 149, 82),
        (156, 130, 68),
        (156, 130, 68),
        (136, 109, 52),
        (136, 109, 52),
        (105, 77, 28),
        (105, 77, 28),
        (233, 229, 223),
        (233, 229, 223),
        (255, 255, 255),
        (255, 255, 255),
    ],
)

CyclicGreen = ColorMap(pos=[0, 0.5, 1], color=["#FFFFFF", "#268086", "#FFFFFF"])


@jit(cache=True)
def create_bmu_composition(bmu_id_for_datapoint, class_as_id, num_bmus, num_bins):
    occurances = np.zeros((num_bmus, num_bins), dtype=np.int32)

    for unique_color in prange(num_bins):
        mask = class_as_id == unique_color

        group_values = bmu_id_for_datapoint[mask]
        occurances[:, unique_color] = np.bincount(group_values, minlength=num_bmus)[
            np.newaxis, ...
        ]

    return occurances


def min_max(array):
    minimum = np.min(array)
    maximum = np.max(array)

    return (array - minimum) / (maximum - minimum), minimum, maximum
