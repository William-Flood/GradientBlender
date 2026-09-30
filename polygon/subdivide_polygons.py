from regions.region import Region
from polygon.smooth import smooth_region_polygons
import numpy as np

def subdivide(regions: list[Region], subdivisions, mesh_smooth_convergence_ratio):
    for region in regions:
        new_polygons = []
        for polygon in region.polygons:
            interpolation_fractions = np.arange(subdivisions + 1) / subdivisions
            opposing_interpolation_fractions = 1 - interpolation_fractions
            top_interpolations = polygon[:, [0]] * opposing_interpolation_fractions[np.newaxis, :] + \
                                 polygon[:, [1]] * interpolation_fractions[np.newaxis, :]
            bottom_interpolations = polygon[:, [3]] * opposing_interpolation_fractions[np.newaxis, :] + \
                                 polygon[:, [2]] * interpolation_fractions[np.newaxis, :]
            row_interpolation_fractions = np.broadcast_to(
                interpolation_fractions[np.newaxis, :], [2, subdivisions + 1]
            )
            row_opposing_interpolation_fractions = 1 - row_interpolation_fractions
            interpolated_grid = top_interpolations[:, np.newaxis, :] * \
                                row_opposing_interpolation_fractions[:, :, np.newaxis] + \
                bottom_interpolations[:, np.newaxis, :] * row_interpolation_fractions[:, :, np.newaxis]
            new_polygons.extend([
                interpolated_grid[:, [tsub, tsub, tsub + 1, tsub + 1], [bsub, bsub + 1, bsub + 1, bsub]].astype(np.int32)
                for tsub in range(subdivisions) for bsub in range(subdivisions)
            ])
        region.polygons = new_polygons
    smooth_region_polygons(regions, mesh_smooth_convergence_ratio)
