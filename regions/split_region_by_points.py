import numpy as np
from regions.identify_regions import identify_regions


def get_division_segment_points(inside_regions, image_shape):
        inside_region_centers_unsorted = np.array([
            np.average(region_points, axis=1)
            for region_points in inside_regions
        ])
        sort_determinant = np.max(inside_region_centers_unsorted, axis=0) - np.min(inside_region_centers_unsorted, axis=0)
        should_sort_vertically = sort_determinant[0] > sort_determinant[1]
        if should_sort_vertically:
            inside_region_sort = np.argsort(inside_region_centers_unsorted[:, 0])
        else:
            inside_region_sort = np.argsort(inside_region_centers_unsorted[:, 1])
        inside_region_centers = inside_region_centers_unsorted[inside_region_sort, :]
        if should_sort_vertically:
            division_points = np.concat([
                [[0, inside_region_centers[1, 0]]],
                inside_region_centers,
                [[image_shape[0], inside_region_centers[-1, 1]]]
            ], axis=0)
        else:
            division_points = np.concat([
                [[inside_region_centers[0, 0], 0]],
                inside_region_centers,
                [[inside_region_centers[-1, 0], image_shape[1]]]
            ], axis=0)
        return division_points, should_sort_vertically


def find_interior_regions(connected_components_points, target_region_points, image_shape):
    components_map = np.zeros(image_shape, dtype=np.int32)
    for component_index, points in enumerate(connected_components_points):
        components_map[*points] = component_index
    target_region_label = components_map[*target_region_points[:, 0]]
    outside_target_region_label = components_map[0, 0]
    if outside_target_region_label == target_region_label:
        outside_edge_ids = np.concat([
            components_map[0, :],
            components_map[:, 0],
            components_map[-1, :],
            components_map[:, -1]
        ])
        edge_ids_not_target = np.logical_not(np.equal(outside_edge_ids, target_region_label))
        if edge_ids_not_target.any():
            outside_target_region_label = outside_edge_ids[np.argmax(edge_ids_not_target)]
    return [connected_component_points
            for component_label, connected_component_points in enumerate(connected_components_points)
            if component_label not in (target_region_label, outside_target_region_label)
            ]


def find_hole_points(points, image_shape):
    inside_outside_map = np.ones(image_shape, dtype=np.bool)
    inside_outside_map[*points] = False
    connected_components_points = identify_regions(inside_outside_map)
    interior_region_points = find_interior_regions(connected_components_points, points, image_shape)
    return interior_region_points


def split_region(points, image_shape):
    inside_regions = find_hole_points(points, image_shape)
    if len(inside_regions) == 0:
        return [points]
    else:
        division_points, should_sort_vertically = get_division_segment_points(
            inside_regions,
            image_shape)
        region_points_with_z = np.pad(points, [[0, 1], [0, 0]]).T
        division_points_with_z = np.pad(division_points, [[0, 0], [0, 1]])
        positive_side_points = []
        negative_side_points = []
        for division_segment_start, division_segment_end in zip(division_points_with_z[:-1], division_points_with_z[1:]):
            segment_vector = division_segment_end - division_segment_start
            if should_sort_vertically:
                points_at_or_after_start = points[0] >= division_segment_start[0]
                points_before_end = points[0] < division_segment_end[0]
            else:
                points_at_or_after_start = points[1] >= division_segment_start[1]
                points_before_end = points[1] < division_segment_end[1]
            points_in_segment_range = region_points_with_z[
                points_at_or_after_start & points_before_end
            ]
            adjusted_region_points_with_z = points_in_segment_range - division_segment_start[np.newaxis, :]
            points_crossed_with_segment = np.cross(segment_vector, adjusted_region_points_with_z)
            points_crossed_with_segment_zs = points_crossed_with_segment[:, 2]
            indices_in_range = np.flatnonzero(points_at_or_after_start & points_before_end)
            positive_side_indices = indices_in_range[np.flatnonzero(points_crossed_with_segment_zs >= 0)]
            negative_side_indices = indices_in_range[np.flatnonzero(points_crossed_with_segment_zs < 0)]
            positive_side_points.append(points[:, positive_side_indices])
            negative_side_points.append(points[:, negative_side_indices])
        return [np.concat(positive_side_points, axis=1), np.concat(negative_side_points, axis=1)]
