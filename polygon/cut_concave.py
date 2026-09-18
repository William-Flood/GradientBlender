from regions.region import Region
import numpy as np
from numpy.typing import NDArray
from util.inspection_utils import test_region
from matplotlib import path
from polygon.polygon_ideal_cuts import find_ideal_cuts
from polygon.cut_polygon import cut_polygon
from polygon.mesh import find_polygons_on_edge
from util.rasterize import rasterize_line, rasterize_polygon


def get_corner_zs(polygon):
    polygon_edges = polygon - np.roll(polygon, 1, axis=1)
    unit_edges = polygon_edges / np.linalg.norm(polygon_edges, axis=0, keepdims=True)
    unit_3_edges = np.pad(unit_edges,((0, 1), (0, 0)), mode="constant", constant_values=0)
    return np.cross(
        unit_3_edges.T,
        np.roll(unit_3_edges, -1, axis=1).T
    )[:, 2]


def check_cut_inside_polygon(polygon, cuts, force_fit):
    """
    Checks which of a series of cuts, if any, is bounded within the given polygon.  Note: this function only checks the
    midpoint of the cuts, and is intended to be used in combination with check_cut_nonintersecting to ensure that the
    cut is well-formed
    Args:
        polygon:
        cuts:
        force_fit:

    Returns:

    """
    polygon_to_list = [(point[0], point[1]) for point in polygon.T]
    polygon_path = path.Path(polygon_to_list)
    cut_middles = np.average(polygon[:, cuts], axis=2)
    is_inside_polygon = polygon_path.contains_points(cut_middles.T)
    radius = 1
    while not is_inside_polygon.any() and force_fit:
        is_inside_polygon = polygon_path.contains_points(cut_middles.T, radius=radius)
        radius += 1
    return is_inside_polygon


def check_cut_nonintersecting(polygon, cuts):
    """
    Detects which cut between polygon vertices crosses one of the edges of the polygon
    Args:
        polygon: A 2xn array defining the vertices of a polygon
        cuts: An nx2 array specifying vertexes as indices of polygon to cut

    Returns: A boolean mask where True indicates a valid cut

    """
    x1 = polygon[1, cuts[:, 0], np.newaxis]
    x2 = polygon[1, cuts[:, 1], np.newaxis]
    y1 = polygon[0, cuts[:, 0], np.newaxis]
    y2 = polygon[0, cuts[:, 1], np.newaxis]
    x3 = polygon[np.newaxis, 1]
    y3 = polygon[np.newaxis, 0]
    rolled_polygon = np.roll(polygon, 1, axis=1)
    x4 = rolled_polygon[np.newaxis, 1]
    y4 = rolled_polygon[np.newaxis, 0]

    t = np.divide(
        np.multiply((x1 - x3), (y3 - y4)) - np.multiply((y1 - y3), (x3 - x4)),
        np.multiply((x1 - x2), (y3 - y4)) - np.multiply((y1 - y2), (x3 - x4))
    )

    u = np.divide(
        np.multiply((y1 - y2), (x1 - x3)) - np.multiply((x1 - x2), (y1 - y3)),
        np.multiply((x1 - x2), (y3 - y4)) - np.multiply((y1 - y2), (x3 - x4))
    )
    intersection_in_cut = np.greater(t, 0) & np.less(t, 1)
    intersection_in_edge = np.greater(u, 0) & np.less(u, 1)
    are_nonintersecting_cuts = np.logical_not(
        np.any(
            intersection_in_cut & intersection_in_edge,
            axis=1
        )
    )
    return are_nonintersecting_cuts


def check_cut(polygon, cuts, force_fit):
    is_cut_nonintersecting = check_cut_nonintersecting(polygon, cuts)
    is_inside = check_cut_inside_polygon(polygon, cuts, force_fit)
    are_valid_cuts = is_cut_nonintersecting & is_inside
    return are_valid_cuts


def rank_candidate_to_other_cuts(is_candidate, polygon_test_values, candidate_vertices, concavity_degree_ranking):
    is_not_candidate = np.logical_not(is_candidate)
    noncandidate = np.flatnonzero(is_not_candidate)

    concave_to_nonconcave_test_values = polygon_test_values[:, is_not_candidate]
    return np.array([
        [candidate_vertices[ranking_1], noncandidate[ranking_2]]
        for ranking_1 in concavity_degree_ranking
            for ranking_2 in np.argsort(concave_to_nonconcave_test_values[ranking_1])
    ])


def rank_candidate_cuts(polygon_test_values, candidate_vertices, concavity_degree_ranking):
        candidate_to_candidate = polygon_test_values[:, candidate_vertices]
        candidate_to_candidate_ranking_unfiltered = np.array([
            [candidate_vertices[ranking_1], candidate_vertices[ranking_2]]
            for ranking_1 in concavity_degree_ranking
                for ranking_2 in np.argsort(candidate_to_candidate[ranking_1]) if ranking_2 > ranking_1
        ])
        return candidate_to_candidate_ranking_unfiltered[
            np.not_equal(
                candidate_to_candidate_ranking_unfiltered[:, 0],
                candidate_to_candidate_ranking_unfiltered[:, 1]
            )
        ]


def rank_evenness_preserving_cuts(connections_2):
    cut_vertex_separation = connections_2[:, 0] - connections_2[:, 1]
    does_cut_preserve_evenness = np.equal(cut_vertex_separation % 2, 1)
    return np.concat([
        connections_2[does_cut_preserve_evenness],
        connections_2[np.logical_not(does_cut_preserve_evenness)]
    ])


def get_test_values(polygon, candidate_vertices):
    candidate_vertex_ideal_cut_unit_vector = find_ideal_cuts(polygon)[:, candidate_vertices]
    polygon_cut_vectors = polygon[:, np.newaxis, :] - polygon[:, candidate_vertices, np.newaxis]
    polygon_cut_distances = np.linalg.norm(polygon_cut_vectors, axis=0, keepdims=True)
    polygon_cut_distances[0, np.arange(candidate_vertices.shape[0]), candidate_vertices] = None
    polygon_cut_unit_vectors = np.divide(polygon_cut_vectors, polygon_cut_distances)
    polygon_cut_dots = np.sum(
        np.multiply(
            candidate_vertex_ideal_cut_unit_vector[:, :, np.newaxis], polygon_cut_unit_vectors
        ),
        axis=0
    )
    return np.multiply(polygon_cut_distances[0], (1 - polygon_cut_dots))


def find_best_cut_between_existing(polygon, candidate_vertices, cross_zs):
    vertex_count = polygon.shape[1]
    concavity_degree_ranking = np.argsort(cross_zs[candidate_vertices])
    polygon_test_values = get_test_values(polygon, candidate_vertices)
    is_concave = np.zeros([vertex_count])
    is_concave[candidate_vertices] = True
    candidate_to_noncandidate_ranking = rank_candidate_to_other_cuts(
        is_concave,
        polygon_test_values,
        candidate_vertices,
        concavity_degree_ranking
    )

    if candidate_vertices.shape[0] == 1 or (candidate_vertices.shape[0] == 2 and max(candidate_vertices) - min(candidate_vertices) == 1):
        connections_1 = candidate_to_noncandidate_ranking
    elif (polygon.shape[1] % 2) == 0 and polygon.shape[1] / 2 == candidate_vertices.shape[0]:
        candidate_to_candidate_ranking_1 = rank_candidate_cuts(
            polygon_test_values, candidate_vertices, concavity_degree_ranking
        )
        alternate_candidate_vertex_map = np.ones(polygon.shape[1], dtype=bool)
        alternate_candidate_vertex_map[candidate_vertices] = False
        alternate_candidate_vertices = np.flatnonzero(alternate_candidate_vertex_map)
        alternate_concavity_degree_ranking = np.argsort(cross_zs[alternate_candidate_vertices])
        alternate_test_values = get_test_values(polygon, alternate_candidate_vertices)
        candidate_to_candidate_ranking_2 = rank_candidate_cuts(
            alternate_test_values, alternate_candidate_vertices, alternate_concavity_degree_ranking
        )
        connections_1 = np.concat([
            candidate_to_candidate_ranking_1, candidate_to_candidate_ranking_2, candidate_to_noncandidate_ranking
        ], axis=0)
    else:
        candidate_to_candidate_ranking = rank_candidate_cuts(polygon_test_values, candidate_vertices, concavity_degree_ranking)
        connections_1 = np.concat([candidate_to_candidate_ranking, candidate_to_noncandidate_ranking], axis=0)
    vertex_index_distance = np.abs(connections_1[:, 0] - connections_1[:, 1])
    connections_2 = connections_1[np.logical_not(np.isin(vertex_index_distance, [0, 1, vertex_count - 1]))]
    if polygon.shape[1] % 2 == 0:
        final_connection_list = rank_evenness_preserving_cuts(connections_2)
    else:
        final_connection_list = connections_2
    are_valid_cuts = check_cut(polygon, final_connection_list, True)
    assert are_valid_cuts.any()
    selected_cut = final_connection_list[np.argmax(are_valid_cuts)]
    return selected_cut


def subdivide_edge(polygon, edge):
    return (polygon[:, (edge + 1) % polygon.shape[1]] + polygon[:, edge]) // 2


def find_best_cut_to_edge(polygon, edge, candidate_vertices):
    vertex_count = polygon.shape[1]
    edge_vector = polygon[:, (edge + 1) % polygon.shape[1]] - polygon[:, edge]
    edge_unit_vector: NDArray = edge_vector / np.linalg.norm(edge_vector)
    edge_cut_point = subdivide_edge(polygon, edge)
    polygon_cut_vectors = polygon[:, candidate_vertices] - edge_cut_point[:, np.newaxis]
    polygon_cut_distances = np.linalg.norm(polygon_cut_vectors, axis=0, keepdims=True)
    polygon_cut_unit_vectors = np.divide(polygon_cut_vectors, polygon_cut_distances)
    polygon_cut_dots = np.sum(
        np.multiply(
            edge_unit_vector[:, np.newaxis], polygon_cut_unit_vectors
        ),
        axis=0
    )
    cut_test_values = np.abs(polygon_cut_dots)
    ordered_vertices = candidate_vertices[np.argsort(cut_test_values)]
    edge_subdivide_vertex = edge+1
    edge_subdivided_polygon = np.insert(polygon, edge_subdivide_vertex, edge_cut_point, axis=1)
    # The typing specified for np.where isn't correct for when the second and third parameters are supplied;
    # wrapping stops the typechecker from complaining
    candidate_vertices_in_new = np.array(
        np.where(
            ordered_vertices < edge_subdivide_vertex,
            ordered_vertices,
            ordered_vertices + 1
        )
    )
    connection_list_1 = np.pad(
        candidate_vertices_in_new[:, np.newaxis],
        ((0, 0), (1, 0)),
        mode="constant",
        constant_values=edge_subdivide_vertex
    )
    vertex_index_distance = np.abs(connection_list_1[:, 0] - connection_list_1[:, 1])
    connection_list_2 = connection_list_1[np.logical_not(
        np.isin(vertex_index_distance, [0, 1, vertex_count - 1])
    )]
    if connection_list_2.shape[0] > 0:
        connection_list_3 = rank_evenness_preserving_cuts(connection_list_2)
        are_valid_cuts = check_cut(edge_subdivided_polygon, connection_list_3, False)
        if are_valid_cuts.any():
            selected_cut = connection_list_3[np.argmax(are_valid_cuts)]
            return edge_subdivided_polygon, selected_cut
        else:
            return None, None
    else:
        return None, None


def choose_cut(polygon, edge_subdivide_ratio, cross_zs):
    cut = None
    edge_cut_data = None
    ups = np.greater(cross_zs, 0)
    downs = np.logical_not(ups)
    num_ups = np.sum(ups)
    num_downs = np.sum(downs)
    if num_ups > num_downs:
        concave_ids = np.flatnonzero(downs)
        adjusted_zs = cross_zs
    elif num_ups == num_downs and np.sum(cross_zs[ups]) > -1 * np.sum(cross_zs[downs]):
        concave_ids = np.flatnonzero(downs)
        adjusted_zs = cross_zs
    else:
        concave_ids = np.flatnonzero(ups)
        adjusted_zs = -1 * cross_zs
    # if polygon.shape[1] % 2 == 1:
    edges = np.roll(polygon, -1, axis=1) - polygon
    edge_lengths = np.linalg.norm(edges, axis=0)
    min_length = min(edge_lengths)
    max_length = max(edge_lengths)
    if max_length / min_length > edge_subdivide_ratio:
        edge = np.argmax(edge_lengths)
        if cross_zs[edge] * cross_zs[(edge + 1) % polygon.shape[1]] > 0:
            candidate_vertices = np.flatnonzero(np.less_equal(cross_zs * cross_zs[edge], 0))
        else:
            candidate_vertices = concave_ids
        subdivided_polygon, cut = find_best_cut_to_edge(polygon, edge, candidate_vertices)
        if subdivided_polygon is not None:
            original_edge = np.stack([polygon[:, edge], polygon[:, (edge + 1) % polygon.shape[1]]], axis=1)
            edge_cut_data = (subdivided_polygon, original_edge)
    if cut is None:
        cut = find_best_cut_between_existing(polygon, concave_ids, adjusted_zs)
    return cut, edge_cut_data


def make_cut(
        polygon,
        cut,
        edge_cut_data,
        remaining_list,
        result_list,
        remaining_list_region,
        polygon_region
):
    if edge_cut_data is None:
        cut_results = cut_polygon(polygon, cut)
        remaining_list.extend(cut_results)
        remaining_list_region.extend([polygon_region] * len(cut_results))
    else:
        polygon_indices, edge_indices = find_polygons_on_edge(
            remaining_list + result_list,
            edge_cut_data[1]
        )
        for polygon_index, edge_index in zip(polygon_indices, edge_indices):
            if len(remaining_list) > polygon_index:
                polygon_to_subdivide = remaining_list[polygon_index]
                subdivision_point = subdivide_edge(polygon_to_subdivide, edge_index)
                new_polygon = np.insert(polygon_to_subdivide, edge_index + 1, subdivision_point, axis=1)
                remaining_list[polygon_index] = new_polygon
            else:
                polygon_to_subdivide = result_list[polygon_index - len(remaining_list)]
                subdivision_point = subdivide_edge(polygon_to_subdivide, edge_index)
                new_polygon = np.insert(polygon_to_subdivide, edge_index + 1, subdivision_point, axis=1)
                result_list[polygon_index - len(remaining_list)] = new_polygon
            assert np.all(np.equal(subdivision_point, np.average(edge_cut_data[1], axis=1).astype(np.int32)))
        cut_results = cut_polygon(edge_cut_data[0], cut)
        remaining_list.extend(cut_results)
        remaining_list_region.extend([polygon_region] * len(cut_results))


def cut_concave_regions(regions: list[Region], edge_subdivide_ratio):
    non_transparent_regions = [region for region in regions if region.value != -1]
    remaining_list = [polygon for region in non_transparent_regions for polygon in region.polygons]
    remaining_list_region = [region_id for region_id, region in enumerate(non_transparent_regions) for _ in region.polygons]
    cut_results = []
    cut_results_region = []
    cut_iteration = 0
    test_rasterization = []
    while len(remaining_list) > 0:
        polygon = remaining_list.pop()
        polygon_region = remaining_list_region.pop()
        region_corner_zs = get_corner_zs(polygon)
        ups = np.greater(region_corner_zs, 0)
        downs = np.logical_not(ups)
        if ups.all() or downs.all():
            cut_results.append(polygon)
            cut_results_region.append(polygon_region)
        else:
            cut, edge_cut_data = choose_cut(polygon, edge_subdivide_ratio, region_corner_zs)
            make_cut(polygon, cut, edge_cut_data, remaining_list, cut_results, remaining_list_region, polygon_region)
        # if cut_iteration % 2 == 0 and cut_iteration > 335:
        # if cut_iteration > 332:
        #     test_rasterization.append(([2000, 3000], np.concat([rasterize_polygon(polygon) for polygon in remaining_list + cut_results], axis=1)))
        #     test_region([2000, 3000], np.concat([rasterize_polygon(polygon) for polygon in remaining_list + cut_results], axis=1))
        cut_iteration += 1
    new_region_polygon_lists = [[]] * len(non_transparent_regions)
    for polygon, region_id in zip(cut_results, cut_results_region):
        new_region_polygon_lists[region_id].append(polygon)
    for region, polygon_list in zip(non_transparent_regions, new_region_polygon_lists):
        region.polygons = polygon_list