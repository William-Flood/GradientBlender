import numpy as np


def cut_polygon(polygon, cut):
    first_vertex, second_vertex = np.sort(cut)
    wrapped_polygon = np.concat([polygon] * 2, axis=1)
    first_cut = wrapped_polygon[:, first_vertex:(second_vertex + 1)]
    second_cut = wrapped_polygon[:, second_vertex:(first_vertex + polygon.shape[1] + 1)]
    return [first_cut, second_cut]