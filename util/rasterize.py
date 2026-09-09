import numpy as np


def rasterize_line(point_1, point_2):
    if np.all(np.equal(point_1, point_2)):
        return point_1[:, np.newaxis]
    point_matrix = np.stack([point_1, point_2], axis=0)
    is_vertical = (
        np.absolute(point_matrix[1, 0] - point_matrix[0, 0]) >
        np.absolute(point_matrix[1, 1] - point_matrix[0, 1])
    )
    if is_vertical:
        cut_point_ordering = np.argsort(point_matrix[:, 0])
        normalized_endpoints = point_matrix[cut_point_ordering]
    else:
        cut_point_ordering = np.argsort(point_matrix[:, 1])
        normalized_endpoints = np.roll(point_matrix[cut_point_ordering], 1, axis=1)
    new_points_normalized_indices = np.arange(normalized_endpoints[1, 0] - normalized_endpoints[0, 0] + 1)
    new_points_normalized_ys = new_points_normalized_indices + normalized_endpoints[0, 0]
    normalized_horizontal_step = (normalized_endpoints[1, 1] - normalized_endpoints[0, 1]) / \
                                 (normalized_endpoints[1, 0] - normalized_endpoints[0, 0])
    new_points_normalized_xs = (
            new_points_normalized_indices * normalized_horizontal_step + normalized_endpoints[0, 1]
    )
    if is_vertical:
        vertiality_oriented_line = np.stack([
            new_points_normalized_ys, new_points_normalized_xs
        ], axis=0)
    else:
        vertiality_oriented_line = np.stack([
            new_points_normalized_xs, new_points_normalized_ys
        ], axis=0)
    if cut_point_ordering[0] == 0:
        return vertiality_oriented_line.astype(np.int32)
    else:
        return np.flip(vertiality_oriented_line, axis=1).astype(np.int32)


def rasterize_polygon(polygon_vertices):
    wrapped_vertices = np.concat([polygon_vertices, polygon_vertices[:, [0]]], axis=1)
    return np.concat(
                [
                    rasterize_line(point_1, point_2) for point_1, point_2 in
                    zip(wrapped_vertices[:, :-1].T, wrapped_vertices[:, 1:].T)
                ], axis=1
            )
