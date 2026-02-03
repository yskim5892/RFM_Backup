import numpy as np

def label_to_rgb(label: np.ndarray) -> np.ndarray:
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, int(label.max()) + 1):
        rgb[label == i] = palette[(i - 1) % len(palette)]
    return rgb

