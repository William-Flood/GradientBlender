import numpy as np
from typing import List
from borders.loop import Loop
from scipy.spatial.distance import cdist
import sys
if "borders.border" not in sys.modules:
    from borders.border import Border
if "borders.loop" not in sys.modules:
    from borders.loop import Loop


def find_closest_points(points_one, points_two):
    point_distances = cdist(points_one, points_two)
    min_point = np.argmin(point_distances)
    return [min_point // points_two.shape[1], min_point % points_two.shape[1]], \
        point_distances[min_point // points_two.shape[1], min_point % points_two.shape[1]]
