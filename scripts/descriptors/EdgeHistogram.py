import cv2
import numpy as np


class EdgeHistogramDescriptor:
    def __init__(self, n_blocks_x=4, n_blocks_y=4, threshold=10):
        self.n_blocks_x = n_blocks_x
        self.n_blocks_y = n_blocks_y
        self.threshold = threshold
        self.edge_types = 5  # Vertical, Horizontal, 45, 135, Non-directional

        # Filter def
        # Vertical
        self.filter_ver = np.array([[1, -1], [1, -1]])
        # Horizontal
        self.filter_hor = np.array([[1, 1], [-1, -1]])
        # 45 deg
        self.filter_45 = np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]])
        # 135 deg
        self.filter_135 = np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]])
        # Non-directional
        self.filter_nondir = np.array([[2, -2], [-2, 2]])

    def describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        h, w = gray.shape
        block_h = int(h / self.n_blocks_y)
        block_w = int(w / self.n_blocks_x)

        feature_vector = []

        resp_ver = cv2.filter2D(gray, -1, self.filter_ver)
        resp_hor = cv2.filter2D(gray, -1, self.filter_hor)
        resp_45 = cv2.filter2D(gray, -1, self.filter_45)
        resp_135 = cv2.filter2D(gray, -1, self.filter_135)
        resp_non = cv2.filter2D(gray, -1, self.filter_nondir)

        responses = np.stack([
            np.abs(resp_ver),
            np.abs(resp_hor),
            np.abs(resp_45),
            np.abs(resp_135),
            np.abs(resp_non)
        ], axis=-1)

        max_responses = np.max(responses, axis=-1)
        dominant_directions = np.argmax(responses, axis=-1)

        for i in range(self.n_blocks_y):
            for j in range(self.n_blocks_x):
                y_start, y_end = i * block_h, (i + 1) * block_h
                x_start, x_end = j * block_w, (j + 1) * block_w

                block_dirs = dominant_directions[y_start:y_end, x_start:x_end]
                block_mags = max_responses[y_start:y_end, x_start:x_end]

                hist_local = np.zeros(self.edge_types)

                valid_edges = block_dirs[block_mags > self.threshold]

                if len(valid_edges) > 0:
                    counts = np.bincount(valid_edges, minlength=5)
                    hist_local = counts.astype(np.float32)

                total = np.sum(hist_local)
                if total > 0:
                    hist_local /= total

                feature_vector.extend(hist_local)

        return np.array(feature_vector)