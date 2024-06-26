import numpy as np

sigmoid = lambda x: 1 / (1 + np.e ** (-1 * x))
relu = lambda x: [x * 0.01, x][int(x > 0)]


class NN:
    """src: https://youtu.be/w8yWXqWQYmU?si=wuQvsusWI7YD3R3q"""

    def __init__(
        self, input_size: int, activation_func, number_of_labyers: int = 1
    ) -> None:
        pass
