import math

import numpy as np
from numpy.typing import NDArray
from regions.identify_regions_floodfill import identify_regions
from util.point_neighborhood import get_neighbor_values_over_array
from scipy.sparse import coo_array
from scipy.spatial.distance import cdist
from util.inspection_utils import *
from util.inspection_utils_gui import visualize_point_list
from util.config import config_obj
from util.rasterize import rasterize_line


def get_region_border(region, target_region, image_shape):
    region_point_map = np.zeros(image_shape, dtype=bool)
    region_point_map[*target_region] = True
    region_point_adjacency = get_neighbor_values_over_array(
        region_point_map,
        region
    )
    contacts_region = np.any(region_point_adjacency, axis=1)
    return region[:, contacts_region]


def find_default_connection_points_and_distances_between(
        region_one,
        region_two
):
    point_distances = cdist(region_one.T, region_two.T, "euclidean").flatten()
    smallest_distance_index = np.argmin(point_distances)
    shortest_distance_points = np.unravel_index(
        smallest_distance_index, [region_one.shape[1], region_two.shape[1]
                                  ])
    return (
        region_one[:, shortest_distance_points[0]],
        region_two[:, shortest_distance_points[1]],
        point_distances[smallest_distance_index]
    )


def assemble_connection_points_and_distances(connection_points_and_distances, region_list):
    one_to_two_indices = np.array([[point_distance_data[3], point_distance_data[4]]
                          for point_distance_data in connection_points_and_distances]).T
    two_to_one_indices = np.array([[point_distance_data[4], point_distance_data[3]]
                          for point_distance_data in connection_points_and_distances]).T
    default_connection_distances = np.zeros([len(region_list)] * 2)
    default_connection_distances[*one_to_two_indices] = [point_distance_data[2]
                          for point_distance_data in connection_points_and_distances]
    default_connection_distances[*two_to_one_indices] = [point_distance_data[2]
                          for point_distance_data in connection_points_and_distances]
    default_connection_distances[
       *np.array([[region_id, region_id] for region_id in range(len(region_list))]).T
    ] = np.max(default_connection_distances) + 1
    default_connection_points = np.zeros([*[len(region_list)] * 2,2])
    default_connection_points[*one_to_two_indices] = [point_distance_data[0]
                          for point_distance_data in connection_points_and_distances]
    default_connection_points[*two_to_one_indices] = [point_distance_data[1]
                          for point_distance_data in connection_points_and_distances]
    return default_connection_distances, default_connection_points


def find_default_connection_points_and_distances(region_list):
    default_connection_points_and_distances = [
            (*find_default_connection_points_and_distances_between(
                region_list[region_one_index],
                region_list[region_two_index]
            ), region_one_index, region_two_index)
            for region_one_index in range(len(region_list) - 1)
            for region_two_index in range(region_one_index + 1, len(region_list))
        ]
    return assemble_connection_points_and_distances(default_connection_points_and_distances, region_list)


def exponential_scale(dots_to_deviations, penalty_scale):
    e = penalty_scale - 0.5
    d = config_obj.cut_direction_change_dropoff_rage
    c = ((1 - d) / (d - e)) ** 2
    b = (e - 1) / (1 - c)
    a = e - b
    f = math.log(c)
    return a + b * np.exp(f * dots_to_deviations)



def directional_weights(dots_to_deviations, penalty_scale):
    linear_scale = (1 - dots_to_deviations) / 2
    exponential_scale_results = exponential_scale(dots_to_deviations, penalty_scale)
    return linear_scale + exponential_scale_results


def test_weights():
    return directional_weights(1 - np.arange(101) / 50, 100)


def get_region_2_point(cut_end, region_points, last_unit_ray, sideways_cut_penalty):
    deviations_from_cut = region_points - cut_end[:, np.newaxis]
    point_distances = np.linalg.norm(deviations_from_cut, axis=0)
    unit_deviations = deviations_from_cut / point_distances
    cut_dots = np.dot(last_unit_ray, unit_deviations)
    is_forward = np.greater(cut_dots, 0)
    if is_forward.any():
        filtered_points = region_points[:, np.greater(cut_dots, 0)]
        filtered_dots = cut_dots[np.greater(cut_dots, 0)]
        filtered_distances = point_distances[np.greater(cut_dots, 0)]
        weighted_distances = np.multiply(filtered_distances, np.exp(-1 * filtered_dots * math.log(sideways_cut_penalty)))
        return filtered_points[:, np.argmin(weighted_distances)]
    else:
        return None


def find_modified_connection_points_and_distances_between(
        region_one,
        region_two,
        last_ray_points,
        sideways_cut_penalty
):
    """
    Finds a new pair of potential cut points based on the last cut made
    Args:
        region_one: The first region to cut between
        region_two: The second region to cut between
        last_ray_points: A list containing the first and second points of the last cut

    Returns: A tuple containing the optimal point on region one to cut from, the optimal point on region two to cut to,
    and the length of the cut

    """
    last_ray = last_ray_points[1] - last_ray_points[0]
    last_unit_ray = last_ray / np.linalg.norm(last_ray, axis=0, keepdims=True)
    region_two_selected_point = get_region_2_point(
        last_ray_points[1], region_two, last_unit_ray, sideways_cut_penalty
    )
    if region_two_selected_point is not None:
        region_point_one_deviations = region_one - region_two_selected_point[:, np.newaxis]
        point_distances = np.linalg.norm(region_point_one_deviations, axis=0)
        selected_point_index = np.argmin(point_distances)
        return (
            region_one[:, selected_point_index],
            region_two_selected_point,
            point_distances[selected_point_index]
        )
    else:
        return None, None, None


def select_cut(cut_and_distances, last_ray):
    cut_distances = np.array([cut_and_distance[2] for cut_and_distance in cut_and_distances])
    cut_vectors = np.array([cut_and_distance[1] - cut_and_distance[0] for cut_and_distance in cut_and_distances]).T
    cut_unit_vectors = cut_vectors / cut_distances
    last_ray_unit = (last_ray[1] - last_ray[0]) / np.linalg.norm(last_ray[1] - last_ray[0])
    cut_dots = np.clip(np.dot(last_ray_unit, cut_unit_vectors), -1.0, 1.0)
    weights = directional_weights(cut_dots, np.max(cut_distances))
    weighted_distances = np.multiply(cut_distances, weights)
    return cut_and_distances[np.argmin(weighted_distances)]


def select_region_and_cut(region_list, region_one_index, backtrack_guard, last_ray, sideways_cut_penalty):
    cut_and_distances = [
        [*find_modified_connection_points_and_distances_between(
            region_list[region_one_index],
            other_region,
            last_ray,
            sideways_cut_penalty
        ), other_index] for other_index, other_region in enumerate(region_list)
        if other_index != region_one_index and other_index not in backtrack_guard
    ]
    forward_cut_and_distances = [cut_and_distance for cut_and_distance in cut_and_distances
                                 if cut_and_distance[0] is not None]
    selected_cut = select_cut(forward_cut_and_distances, last_ray)
    return selected_cut


def rasterize_cut(new_border_start_ends_list):
    new_points_list = []
    new_border_start_ends = np.array(new_border_start_ends_list)
    for cut in new_border_start_ends:
        new_points_list.append(rasterize_line(cut[0], cut[1]))
    new_points = np.concat(new_points_list, axis=1).astype(np.int32)
    return new_points


def find_intra_target_borders(inside_regions, outside_regions, sideways_cut_penalty):
    full_region_list = [*inside_regions, outside_regions]
    is_cut_out = np.array([
        *([False] * len(inside_regions)), True
    ])
    default_connection_distances, default_connection_points = find_default_connection_points_and_distances(
        full_region_list
    )
    new_border_start_ends = []
    traversals = []
    traversals_points = []
    traversal_indices = []
    while not is_cut_out.all():
        cut_out_indices = np.flatnonzero(is_cut_out)
        not_cut_out_indices = np.flatnonzero(np.logical_not(is_cut_out))
        not_cut_out_to_cut_out_indices = np.stack(np.broadcast_arrays(
            not_cut_out_indices[:, np.newaxis],
            cut_out_indices[np.newaxis, :]
        ), axis=0)
        not_cut_out_to_cut_out_distances = default_connection_distances[*not_cut_out_to_cut_out_indices]
        cut_to_and_from = np.nonzero(
            np.equal(not_cut_out_to_cut_out_distances, np.min(not_cut_out_to_cut_out_distances))
        )
        cut_traversal_index = not_cut_out_indices[cut_to_and_from[0][0]]
        cut_from_index = cut_out_indices[cut_to_and_from[1][0]]
        last_ray = [
            default_connection_points[cut_from_index, cut_traversal_index],
            default_connection_points[cut_traversal_index, cut_from_index]
        ]
        new_border_start_ends.append(last_ray)
        previous_uncut = []
        traversal = [last_ray]
        traversal_indices.append([cut_from_index, cut_traversal_index])
        while not is_cut_out[cut_traversal_index]:
            next_cut = select_region_and_cut(
                full_region_list,
                cut_traversal_index,
                previous_uncut,
                last_ray,
                sideways_cut_penalty
            )
            previous_uncut.append(cut_traversal_index)
            last_ray = [next_cut[0], next_cut[1]]
            new_border_start_ends.append(last_ray)
            cut_traversal_index = next_cut[3]
            traversal.append(last_ray)
            traversal_indices[-1].append(cut_traversal_index)
        traversals_points.append(rasterize_cut(traversal))
        is_cut_out[previous_uncut] = True
        traversals.append(traversal)
    point_traversal_indices = np.array([traversal_index for traversal_index, traversal_points in enumerate(traversals_points)
                                  for _ in traversal_points.T])
    traversal_points = np.concat(traversals_points, axis=1)
    def visualize_traversal_borders():
        regions_points = np.concat([*inside_regions, outside_regions], axis=1)
        traversal_and_borders_map = np.concat([point_traversal_indices, [-1] * regions_points.shape[1]])
        regions_and_traversals_points = np.concat([traversal_points, regions_points], axis=1)
        point_list, point_indices = np.unique(regions_and_traversals_points, return_index=True, axis=1)
        traversal_map = traversal_and_borders_map[point_indices]
        visualize_point_list(point_list, traversal_map)
    def visualize_region_borders():
        regions_points = np.concat([*inside_regions, outside_regions], axis=1)
        regions_indices = np.concat([
            *list([region_id] * region.shape[1] for region_id, region in enumerate(inside_regions)),
            [len(inside_regions)] * outside_regions.shape[1]
        ])
        visualize_point_list(regions_points, regions_indices)
    return np.unique(traversal_points, axis=1)


def get_split_regions_borders(inside_regions, outside_regions, target_region, image_shape, sideways_cut_penalty):
    if len(outside_regions) == 0:
        outside_points = get_outside_coords(image_shape)
    else:
        outside_points = np.concat([*outside_regions, get_outside_coords(image_shape)], axis=1)
    outside_border = get_region_border(outside_points, target_region, image_shape)
    inside_borders = [get_region_border(region, target_region, image_shape) for region in inside_regions]
    between_borders = find_intra_target_borders(inside_borders, outside_border, sideways_cut_penalty)
    return between_borders


def get_outside_coords(image_shape):
    full_indices = np.reshape(np.indices(image_shape), [2, -1])
    return full_indices[:, (
        np.isin(full_indices[0], [0, image_shape[0] - 1]) |
        np.isin(full_indices[1], [0, image_shape[1] - 1])
    )]


def find_interior_regions(connected_components_points, target_region_points, image_shape):
    components_map = np.full(image_shape, -1, dtype=np.int32)
    for component_index, points in enumerate(connected_components_points):
        components_map[*points] = component_index
    target_region_label = components_map[*target_region_points[:, 0]]
    outside_coords = get_outside_coords(image_shape)

    outside_target_indices = outside_coords[
        :,
        np.not_equal(components_map[*outside_coords], target_region_label)]
    outside_target_region_labels = np.unique(components_map[*outside_target_indices])
    return [
        connected_component_points
        for component_label, connected_component_points in enumerate(connected_components_points)
        if component_label not in (target_region_label, *outside_target_region_labels)
    ], [
        connected_component_points
        for component_label, connected_component_points in enumerate(connected_components_points)
        if component_label in outside_target_region_labels
    ]


def find_hole_points(points, image_shape):
    inside_outside_map = np.ones(image_shape, dtype=bool)
    inside_outside_map[*points] = False
    connected_components_points = identify_regions(inside_outside_map)
    interior_region_points, outside_region_points = find_interior_regions(connected_components_points, points, image_shape)
    return interior_region_points, outside_region_points


def split_region(points, image_shape, sideways_cut_penalty):
    inside_regions, outside_regions = find_hole_points(points, image_shape)
    if len(inside_regions) == 0:
        return [points]
    else:
        between_borders: NDArray = get_split_regions_borders(
            inside_regions,
            outside_regions,
            points,
            image_shape,
            sideways_cut_penalty
        )
        split_points_mask = np.ones(image_shape, dtype=bool)
        split_points_mask[*np.concat(inside_regions, axis=1)] = False
        split_points_mask[*np.concat(outside_regions, axis=1)] = False
        split_points_mask[*between_borders] = False
        # test_boolean_matrix(split_points_mask)
        new_regions = identify_regions(split_points_mask, split_points_mask, include_diagonals=False)
        new_regions_matrix = np.full(image_shape, -1)
        for region_id, region in enumerate(new_regions):
            new_regions_matrix[*region] = region_id
        points_to_fill: NDArray = between_borders
        while points_to_fill.shape[1] > 0:
            between_borders_neighbors = get_neighbor_values_over_array(new_regions_matrix, points_to_fill)
            border_assignment_indices = np.argmax(np.not_equal(between_borders_neighbors, -1), axis=1, keepdims=True)
            border_assignments = np.take_along_axis(between_borders_neighbors, border_assignment_indices, axis=1)
            new_regions_matrix[*points_to_fill] = border_assignments[:, 0]
            points_to_fill = points_to_fill[:, np.equal(new_regions_matrix[*points_to_fill], -1)]
        return [np.array(
            np.nonzero(
                np.equal(new_regions_matrix, region_id)
            )
        ) for region_id in range(len(new_regions))]