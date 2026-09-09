from regions.region import Region
import numpy as np
from polygon.polygon_ideal_cuts import find_ideal_cuts
from polygon.cut_polygon import cut_polygon


def cut_quad_from_polygon(polygon):
    ideal_cuts = -1 * find_ideal_cuts(polygon)
    if np.isinf(ideal_cuts[0]).any():
        selected_cut = np.flatnonzero(np.isinf(ideal_cuts[0]))[-1]
    else:
        potential_cuts = polygon - np.roll(polygon, 3, axis=1)
        potential_unit_cuts = potential_cuts / np.linalg.norm(potential_cuts, axis=0, keepdims=True)
        cut_check = np.sum(np.multiply(ideal_cuts, potential_unit_cuts), axis=0)
        selected_cut = np.argmax(cut_check)
    return cut_polygon(polygon, [selected_cut, selected_cut + 3])


def cut_out_quads(regions: list[Region]):
    for region in regions:
        polygon_stack = region.polygons.copy()
        new_polygons = []
        while len(polygon_stack) > 0:
            polygon = polygon_stack.pop()
            if polygon.shape[1] <= 4:
                new_polygons.append(polygon)
            else:
                quad, remainder = cut_quad_from_polygon(polygon)
                new_polygons.append(quad)
                polygon_stack.append(remainder)
        region.polygons = new_polygons
