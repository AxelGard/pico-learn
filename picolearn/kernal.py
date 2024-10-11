import numpy as np


def filter(on:np.ndarray, kernal:np.ndarray) -> np.ndarray:
    return np.convolve(on, kernal)
