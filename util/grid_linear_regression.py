import numpy as np

# Unused; retained for future reference
def regress_graph(points_per_grid, padding_mask):
    average_points_in_grids = np.sum(points_per_grid * padding_mask[np.newaxis, :, :], axis=2) / \
                              np.sum(padding_mask, axis=1, keepdims=True).T
    displacements_from_average_in_grids = (points_per_grid - average_points_in_grids[:, :, np.newaxis]) * \
                                          padding_mask[np.newaxis, :, :]
    betas = np.sum(
        displacements_from_average_in_grids[0, :, :] * displacements_from_average_in_grids[1, :, :],
        axis=1
    ) / \
            (np.sum(np.square(displacements_from_average_in_grids[1, :, :]), axis=1) + 1e-8)
    alphas = average_points_in_grids[0, :] - betas * average_points_in_grids[1, :]
    predicted_ys = alphas[:, np.newaxis] + betas[:, np.newaxis] * points_per_grid[1, :, :]
    residual_sum_of_squares = np.sum(
        np.square(
            points_per_grid[0, :, :] - predicted_ys
        ) * padding_mask,
        axis=1
    ) / np.sum(padding_mask, axis=1)
    r_squareds = 1 - residual_sum_of_squares / (
            np.sum(np.square(displacements_from_average_in_grids[0, :, :]) * padding_mask, axis=1) / np.sum(
        padding_mask, axis=1)
            + 1e-8
    )
    return alphas, betas, r_squareds