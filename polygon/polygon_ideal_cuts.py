import numpy as np


def find_ideal_cuts(polygon):
    positive_roll_edge = polygon - np.roll(polygon, 1, axis=1)
    negative_roll_edge = polygon - np.roll(polygon, -1, axis=1)
    positive_roll_unit_edge = positive_roll_edge / np.linalg.norm(positive_roll_edge, axis=0, keepdims=True)
    negative_roll_unit_edge = negative_roll_edge / np.linalg.norm(negative_roll_edge, axis=0, keepdims=True)
    vertex_edge_averages = (positive_roll_unit_edge + negative_roll_unit_edge) / 2
    vertex_ideal_cuts = vertex_edge_averages / np.linalg.norm(vertex_edge_averages, axis=0, keepdims=True)
    return vertex_ideal_cuts
