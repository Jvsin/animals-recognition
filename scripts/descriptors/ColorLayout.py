import cv2
import numpy as np


class ColorLayoutDescriptor:
    def __init__(self, grid_size=(8, 8), Y_coeffs=12, CbCr_coeffs=6):
        self.grid_size = grid_size
        self.Y_coeffs = Y_coeffs
        self.CbCr_coeffs = CbCr_coeffs

        self.scan_order = self._create_scan_order(grid_size)

    @staticmethod
    def _create_scan_order(size):
        rows, cols = size
        indices = []
        for r in range(rows):
            for c in range(cols):
                indices.append((r, c))

        indices.sort(key=lambda x: (x[0] + x[1], x[0] if (x[0] + x[1]) % 2 == 0 else x[1]))
        return indices

    def describe(self, image):
        image_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        ycc_grid = cv2.resize(image_ycc, (self.grid_size[1], self.grid_size[0]), interpolation=cv2.INTER_AREA)

        feature_vector = []

        for i in range(3):
            channel = np.float32(ycc_grid[:, :, i])

            dct_coeffs = cv2.dct(channel)

            if i == 0:
                num_coeffs = self.Y_coeffs
            else:
                num_coeffs = self.CbCr_coeffs

            max_coeffs = self.grid_size[0] * self.grid_size[1]
            num_coeffs = min(num_coeffs, max_coeffs)

            selected_coeffs = []
            for idx, (r, c) in enumerate(self.scan_order):
                if idx >= num_coeffs:
                    break
                selected_coeffs.append(dct_coeffs[r, c])

            feature_vector.extend(selected_coeffs)

        return np.array(feature_vector)
