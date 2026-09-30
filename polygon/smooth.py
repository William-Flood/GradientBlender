from gradengrs import smooth_polygons
from regions.region import Region
import numpy as np

def smooth_region_polygons(regions: list[Region], mesh_smooth_convergence_ratio):
    for region in regions:
        region_polygons = []
        for polygon in region.polygons:
            if polygon.shape[1] > 2 and np.equal(np.unique(polygon, axis=1, return_counts=True)[1], 1).all():
                region_polygons.append(polygon)
        smooth_results = smooth_polygons(region_polygons, mesh_smooth_convergence_ratio)
        region.polygons = [result.points for result in smooth_results]