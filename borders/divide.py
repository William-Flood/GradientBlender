import numpy as np
from scipy.sparse.csgraph import connected_components
from borders.border import Border
from typing import List


def divide_borders(border_points) -> List[Border]:
    connecting_offsets = np.array(
        [[row_offset, column_offset] for column_offset in [-1, 0, 1] for row_offset in [-1, 0, 1]]
    ).T
    border_points_and_connections = border_points[:, np.newaxis, :] + connecting_offsets[:, :, np.newaxis]
    compare_points = np.repeat(border_points[:, np.newaxis, :], 9, axis=1)
    point_comparison = np.equal(border_points_and_connections[:, :, :, np.newaxis], compare_points[:, :, np.newaxis, :])
    border_graph = np.logical_and(
        np.any(
            np.all(
                point_comparison, axis=0
            ),
            axis=0
        ),
        np.logical_not(
            np.identity(border_points.shape[1]), dtype=np.bool
        )
    )
    n_components, labels = connected_components(border_graph)
    split_borders = []
    for split in range(n_components):
        border_segment = border_points[:,
                             np.argwhere(labels == split)[:, 0]
                             ]
        border_segment_filtered = np.unique(border_segment, axis=1)
        split_borders.append(Border(border_segment_filtered))
    return split_borders
