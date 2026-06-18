import numpy as np
from numpy.typing import NDArray


class Border:
    def __init__(self, border_points: NDArray[np.int32]):
            self.full_points = border_points
    decomposed_points = NDArray[np.int32]