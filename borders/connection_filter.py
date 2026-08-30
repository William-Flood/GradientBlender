from PIL import Image
import numpy as np
import os
from collections import defaultdict
from borders.get_pixel_connections import get_pixel_connections
from scipy.sparse import coo_array
from util.point_neighborhood import offset_matrix
from time import time

SECTION_COMBINER = np.array([[2 ** power for power in range(12)]])
EXTENDED_NEIGHBOR_WINDOW = np.reshape(np.indices([5, 5]) - 2, [2, 1, 25])


def hash_neighborhood_vectors(neighborhood_matrix):
    """
    Converts an nx25 matrix into an array of integer values for use in a lookup table
    Args:
        neighborhood_matrix:

    Returns:

    """
    section_values = np.sum(
        np.stack([
            SECTION_COMBINER * neighborhood_matrix[:, :12],
            SECTION_COMBINER * neighborhood_matrix[:, 13:]
        ], axis=0),
    axis=2)
    return (((section_values[0] * 12289) % 6151 + section_values[1]) * 12289) % 6151


class ConnectionFilter:
    """
    Used to inspect a region border by comparing
    """
    def __init__(self, filter_location="filters.png"):
        self.filter_hash_dict = dict()
        self.remaining_branch_neighborhoods = np.zeros([0, 25])
        self.filter_profiling = []
        if os.path.isfile(filter_location):
            filter_matrix = np.array(Image.open(filter_location)).astype(np.bool)[:, :, 0]
            assert filter_matrix.shape[0] % 5 == 0
            filters = np.transpose(np.reshape(filter_matrix, [
                filter_matrix.shape[0] // 5,
                5,
                3,
                5
            ]), [0, 2, 1, 3])
            filters_with_wildcards = np.flatnonzero(
                np.any(
                    np.reshape(filters[:, 1], [-1, 25]), axis=1
                )
            )
            kernels_without_wildcards = np.flatnonzero(
                np.logical_not(np.any(
                    filters[:, 1]
                ))
            )
            full_filter_list = []

            for filter_index in filters_with_wildcards:
                wildcard_indices = np.nonzero(
                    filters[filter_index, 1]
                )
                wildcard_options = np.zeros([1, 5, 5], dtype=np.bool)
                for y, x in np.array(wildcard_indices).T:
                    new_options = np.copy(wildcard_options)
                    new_options[:, y, x] = True
                    wildcard_options = np.concat([wildcard_options, new_options], axis=0)
                new_filters = filters[[filter_index], 0] + wildcard_options
                false_connections_repeated = np.stack([filters[filter_index, 2]] * wildcard_options.shape[0], axis=0)
                full_filter_list.append(np.stack([new_filters, false_connections_repeated], axis=1))
            filters_and_false_connections = filters[:, [0, 2]]
            full_filter_list.append(filters_and_false_connections[kernels_without_wildcards])
            full_orientation_1 = np.concat(full_filter_list, axis=0)
            full_orientation_2 = np.flip(full_orientation_1, axis=2)
            full_orientation_1_2 = np.concat([full_orientation_1, full_orientation_2], axis=0)
            full_filters_with_duplicates = np.reshape(
                np.concat([
                    full_orientation_1_2,
                    np.rot90(full_orientation_1_2, 1, (2, 3)),
                    np.rot90(full_orientation_1_2, 2, (2, 3)),
                    np.rot90(full_orientation_1_2, 3, (2, 3))
                ]), [-1, 2, 25]
            )
            _, full_filter_indices = np.unique(full_filters_with_duplicates, return_index=True, axis=0)
            full_filters = full_filters_with_duplicates[full_filter_indices]
            filter_hashes = hash_neighborhood_vectors(full_filters[:, 0])
            unique_hashes = np.unique(filter_hashes)
            self.filter_keys = unique_hashes
            for kernel_hash in unique_hashes:
                matching_kernels = np.flatnonzero(np.equal(filter_hashes, kernel_hash))
                self.filter_hash_dict[kernel_hash] = full_filters[matching_kernels]
        else:
            self.filter_keys = np.zeros([0])

    def get_filtered_connections(self, points, image_shape):
        start_time = time()
        # Note - these values are offset by 1 in point_index_map to allow sparse storage
        unfiltered_point_connections = get_pixel_connections(points, image_shape)
        adjusted_points = points + 2
        point_index_map = coo_array(
            (np.arange(points.shape[1]) + 1, adjusted_points),
            shape=(np.array(image_shape) + 4)
        )
        point_bool_mask = coo_array(
            ([True] * points.shape[1], adjusted_points),
            shape=(np.array(image_shape) + 4)
        )
        adjusted_immediate_neighborhood = adjusted_points[:, :, np.newaxis] + offset_matrix
        branch_point_indices = np.flatnonzero(
            np.greater(
                np.sum(point_bool_mask[*adjusted_immediate_neighborhood].toarray(), axis=1), 2
            )
        )
        if branch_point_indices.shape[0] == 0:
            self.filter_profiling.append([time() - start_time, 0, 0])
            return unfiltered_point_connections, np.zeros([2, 0], dtype=np.int32)
        branch_points = adjusted_points[:, branch_point_indices]
        branch_point_neighborhoods = branch_points[:, :, np.newaxis] + EXTENDED_NEIGHBOR_WINDOW
        branch_point_neighborhood_bools = point_bool_mask[*branch_point_neighborhoods].toarray()
        neighborhood_hashes = hash_neighborhood_vectors(branch_point_neighborhood_bools)
        unique_point_hashes = np.unique(neighborhood_hashes)
        matching_hashes = unique_point_hashes[np.isin(unique_point_hashes, self.filter_keys)]
        offset_false_connection_indices = []
        for neighborhood_hash in matching_hashes:
            matching_branches = np.flatnonzero(np.equal(neighborhood_hashes, neighborhood_hash))
            point_neighborhoods = branch_point_neighborhood_bools[matching_branches]
            filters = self.filter_hash_dict[neighborhood_hash]
            filter_matches = np.array(
                np.nonzero(
                    np.all(
                        np.equal(point_neighborhoods[:, np.newaxis, :], filters[np.newaxis, :, 0, :]), axis=2
                    )
                )
            )
            for neighborhood_hash_index, filter_index in filter_matches.T:
                index_in_branches = matching_branches[neighborhood_hash_index]
                branch_point_coords = branch_points[
                    :,
                    index_in_branches
                ]
                branch_point_index = point_index_map[*branch_point_coords]
                false_connection_locations = filters[filter_index, 1]
                neighborhood_coords = branch_point_neighborhoods[:, index_in_branches]
                neighborhood_indices = point_index_map[*neighborhood_coords].toarray()
                neighborhood_false_connection_indices = neighborhood_indices[false_connection_locations]
                assert 0 not in neighborhood_false_connection_indices
                offset_false_connection_indices.append(np.stack([
                    [branch_point_index] * neighborhood_false_connection_indices.shape[0],
                    neighborhood_false_connection_indices
                ], axis=0))
        if len(offset_false_connection_indices) == 0:
            self.filter_profiling.append([time() - start_time, branch_point_indices.shape[0], 0])
            return unfiltered_point_connections, points[:, branch_point_indices]
        false_connection_indices_one_way = np.concat(offset_false_connection_indices, axis=1) - 1
        false_connection_indices = np.unique(
            np.concat(
                [
                    false_connection_indices_one_way,
                    np.roll(false_connection_indices_one_way, 1, axis=0)
                ],
                axis=1
            ),
            axis=1
        )
        false_connection_map = coo_array(
            ([True] * false_connection_indices.shape[1], false_connection_indices),
            shape=[points.shape[1]] * 2
        )
        is_false_connection = false_connection_map[*unfiltered_point_connections].toarray()
        remaining_connections = unfiltered_point_connections[:, np.logical_not(is_false_connection)]
        point_indices_in_remaining, point_index_counts = np.unique(remaining_connections[0], return_counts=True)
        remaining_branches = point_indices_in_remaining[np.greater(point_index_counts, 2)]
        if remaining_branches.shape[0] > 0:
            remaining_branch_neighborhoods_this_round = point_bool_mask[*(
                    points[:, remaining_branches, np.newaxis] + EXTENDED_NEIGHBOR_WINDOW + 2
            )].toarray()
            self.remaining_branch_neighborhoods = np.unique(
                np.concat([
                    self.remaining_branch_neighborhoods,
                    remaining_branch_neighborhoods_this_round
                ], axis=0),
                axis=0
            )
        self.filter_profiling.append([time() - start_time, branch_point_indices.shape[0], remaining_branches.shape[0]])
        return remaining_connections, points[:, remaining_branches]

    def save_remaining_samples(self, save_file_name="remaining_branch_samples.png"):
        if self.remaining_branch_neighborhoods.shape[0] > 0:
            shaped_remaining_samples = np.reshape(self.remaining_branch_neighborhoods,[-1, 5])
            Image.fromarray(shaped_remaining_samples.astype(np.uint8) * 255).save(save_file_name)
