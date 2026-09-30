from regions.region import Region
import numpy as np
from polygon.polygon_ideal_cuts import find_ideal_cuts
from polygon.cut_polygon import cut_polygon
from gradengrs import cut_out_trianges
from collections import defaultdict
from polygon.smooth import smooth_region_polygons


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


def cut_down_to_quads(regions: list[Region]):
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



def cut_up_to_quads(regions: list[Region]):
    polygon_list = []
    polygon_region_list = []
    non_transparent_regions = [region for region in regions if region.value != -1]
    for region in non_transparent_regions:
        for polygon in region.polygons:
            if polygon.shape[1] > 2 and np.equal(np.unique(polygon, axis=1, return_counts=True)[1], 1).all():
                polygon_list.append(polygon)
                polygon_region_list.append(region.id)
    cut_results = cut_out_trianges(polygon_list)
    new_region_polygon_matcher = defaultdict(list)
    for cut_result in cut_results:
        new_region_polygon_matcher[polygon_region_list[cut_result.original_id]].append(cut_result.points)
    for region in non_transparent_regions:
        region.polygons = new_region_polygon_matcher[region.id]


def cut_out_quads(regions: list[Region], mesh_smooth_convergence_ratio):
    cut_down_to_quads(regions)
    smooth_region_polygons(regions, mesh_smooth_convergence_ratio)
    cut_up_to_quads(regions)