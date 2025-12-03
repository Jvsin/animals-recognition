import cv2
import numpy as np


class ColorStructureDescriptor:
    def __init__(self, quantization_levels=32, window_size=8):
        self.quantization_levels = quantization_levels
        self.window_size = window_size

        if quantization_levels == 32:
            self.h_bins, self.s_bins, self.v_bins = 8, 2, 2
        elif quantization_levels == 64:
            self.h_bins, self.s_bins, self.v_bins = 8, 4, 2
        elif quantization_levels == 128:
            self.h_bins, self.s_bins, self.v_bins = 16, 4, 2
        elif quantization_levels == 256:
            self.h_bins, self.s_bins, self.v_bins = 16, 4, 4
        else:
            cbrt = int(round(quantization_levels ** (1 / 3)))
            self.h_bins = cbrt
            self.s_bins = cbrt
            self.v_bins = quantization_levels // (cbrt * cbrt)

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))

    def describe(self, image):
        target_width = 256
        h, w = image.shape[:2]
        scale = target_width / float(w)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

        h_idx = np.floor((hsv[:, :, 0] / 180.0) * self.h_bins).astype(np.int32)
        s_idx = np.floor((hsv[:, :, 1] / 256.0) * self.s_bins).astype(np.int32)
        v_idx = np.floor((hsv[:, :, 2] / 256.0) * self.v_bins).astype(np.int32)

        h_idx = np.clip(h_idx, 0, self.h_bins - 1)
        s_idx = np.clip(s_idx, 0, self.s_bins - 1)
        v_idx = np.clip(v_idx, 0, self.v_bins - 1)

        quantized_image = (h_idx * (self.s_bins * self.v_bins)) + (s_idx * self.v_bins) + v_idx

        feature_vector = np.zeros(self.quantization_levels, dtype=np.float32)

        for k in range(self.quantization_levels):
            binary_mask = (quantized_image == k).astype(np.uint8)

            if cv2.countNonZero(binary_mask) == 0:
                continue

            dilated_mask = cv2.dilate(binary_mask, self.kernel)

            count = cv2.countNonZero(dilated_mask)
            feature_vector[k] = count

        total_pixels = resized_image.shape[0] * resized_image.shape[1]
        if total_pixels > 0:
            feature_vector /= total_pixels

        return feature_vector