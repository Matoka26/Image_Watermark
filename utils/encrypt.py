import numpy as np


def hi(matrix: np.ndarray):
    val = 8000000
    matrix[50:150, 100] = val
    matrix[50:150, 140] = val
    matrix[100, 100:150] = val
    # I
    matrix[50:150, 200] = val
    matrix[50, 180:230] = val
    matrix[150, 180:230] = val