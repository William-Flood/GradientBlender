import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_array
from util.substitute_values import substitute_values
from util.point_neighborhood import offset_matrix
from skimage.segmentation import flood


def merge_regions(merges_array, regions_array):
    overlaps_unfiltered = np.unique(np.concat(merges_array, axis=1), axis=1)
    overlaps, overlaps_point_indices = np.unique(overlaps_unfiltered, return_inverse=True, axis=1)
    overlaps_unique_regions, overlap_region_indices, overlaps_latent = np.unique(overlaps,
                                                                                 return_index=True,
                                                                                 return_inverse=True)
    overlaps_grid = coo_array(
        (np.ones(overlaps_latent.shape[1], dtype=np.bool), overlaps_latent),
        shape=[np.max(overlaps_latent) + 1] * 2
    )
    _, overlap_groups = connected_components(overlaps_grid)
    _, overlap_group_starts, group_membership = np.unique(overlap_groups, return_index=True, return_inverse=True)
    overlapping_regions_merged = overlaps_unique_regions[overlap_group_starts[group_membership]]
    substitute_values(regions_array, overlaps_unique_regions, overlapping_regions_merged)


def get_region_merges(search_points, regions_array, region_membership):
    search_points_over = regions_array[*search_points]
    is_point_over_regions = np.not_equal(search_points_over, -1)
    return np.unique(
        np.stack(
            (region_membership[is_point_over_regions],
            search_points_over[is_point_over_regions]),
            axis=0
        ),
        axis=1
    )

def filter_boundary_cross(search_points,
                          values_array,
                          regions_array,
                          starting_points,
                          search_point_regions
                          ):
    starting_regions_current_unfiltered = regions_array[*starting_points]
    starting_regions_current, unique_starting, starting_to_region = np.unique(
        starting_regions_current_unfiltered,
        return_index=True,
        return_inverse=True
    )
    current_region_starting_points = starting_points[:, unique_starting]
    starting_values_current = values_array[*current_region_starting_points]
    search_values_over = values_array[*search_points]
    search_values_to_match = np.copy(search_point_regions)
    substitute_values(search_values_to_match, starting_regions_current, starting_values_current)
    filtered_points = search_points[:, np.equal(search_values_over, search_values_to_match)]
    filtered_point_regions = search_point_regions[np.equal(search_values_over, search_values_to_match)]
    return filtered_points, filtered_point_regions


def find_next_points(last_points, regions_array):
    neighborhood_regions = regions_array[*last_points]

    candidate_neighbors = np.clip(
        last_points[:, :, np.newaxis] + offset_matrix,
        np.zeros([2, last_points.shape[1], 8]).astype(np.int32),
        np.stack(
            (
                np.full([last_points.shape[1], 8], regions_array.shape[0]),
                np.full([last_points.shape[1], 8], regions_array.shape[1])
            ),
            axis=0
        ).astype(np.int32) - 1
    )
    candidate_regions = regions_array[*candidate_neighbors]
    candidate_keep = np.not_equal(neighborhood_regions[:, np.newaxis], candidate_regions)
    next_points = np.reshape(candidate_neighbors, [2, -1])[:, candidate_keep.flatten()]
    region_membership = np.repeat(neighborhood_regions[:, np.newaxis], 8, axis=1)[candidate_keep]
    return next_points, region_membership


def search_from_start(starting_points, values_array, regions_array):
    last_points = starting_points
    regions_array[*starting_points] = np.arange(
        np.max(regions_array),
        int(np.max(regions_array)) + starting_points.shape[1]
    ) + 1
    merges = []
    test_i = 0
    while True:
        candidate_next_points, candidate_region_membership = find_next_points(last_points, regions_array)
        next_points, region_membership = filter_boundary_cross(
            candidate_next_points,
            values_array,
            regions_array,
            starting_points,
            candidate_region_membership
        )
        if next_points.shape[1] == 0:
            break
        round_merges = get_region_merges(next_points, regions_array, region_membership)
        if round_merges.shape[1] != 0:
            merges.append(round_merges)
        next_points_merge_filtered = next_points[
            :,
            np.equal(regions_array[*next_points], -1)
        ]
        regions_array[*next_points] = region_membership
        last_points = np.unique(next_points_merge_filtered, axis=1)
        test_i += 1
    if len(merges) > 0:
        merge_regions(merges, regions_array)


rng = np.random.default_rng(19300511)
def make_starting_points(regions_array):
    # May need to be tuned per system specs
    MAX_STARTING_SAMPLES = 100
    regions_points = np.indices(regions_array.shape)
    are_points_open = np.equal(regions_array, -1)
    open_points = regions_points[:, are_points_open]
    sample_point_count = min(int(np.sqrt(open_points.shape[1])), MAX_STARTING_SAMPLES)
    sample_indices = rng.choice(np.arange(open_points.shape[1]), sample_point_count, replace=False)
    return open_points[:, sample_indices]


def identify_regions(values_array, search_mask=None, include_diagonals=True):
    if search_mask is None:
        filled = np.zeros(values_array.shape, dtype=np.bool)
    else:
        filled = np.logical_not(search_mask)
    if include_diagonals:
        connectivity = None
    else:
        connectivity = 1
    regions = []
    while np.logical_not(filled).any():
        next_point = np.unravel_index(
            np.argmin(filled.flatten()), values_array.shape
        )
        region_mask = flood(values_array, next_point, connectivity=connectivity)
        regions.append(np.array(np.nonzero(region_mask)))
        filled = filled + region_mask
    return regions
