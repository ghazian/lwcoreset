import numpy as np

def parse_txt(path) -> np.ndarray:
    data = np.loadtxt(path, dtype=int)
    return data