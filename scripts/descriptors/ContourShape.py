import cv2
import numpy as np


class ContourShapeDescriptor:
    def __init__(self, num_coeffs=32):
        self.num_coeffs = num_coeffs

    def describe(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(self.num_coeffs)

        c = max(contours, key=cv2.contourArea)

        if len(c) < self.num_coeffs:
            return np.zeros(self.num_coeffs)

        contour_array = c[:, 0, :]
        contour_complex = np.empty(contour_array.shape[0], dtype=complex)
        contour_complex.real = contour_array[:, 0]
        contour_complex.imag = contour_array[:, 1]

        fourier_result = np.fft.fft(contour_complex)

        fourier_result[0] = 0

        magnitudes = np.abs(fourier_result)

        if magnitudes[1] > 0:
            magnitudes = magnitudes / magnitudes[1]
        else:
            return np.zeros(self.num_coeffs)

        features = magnitudes[2: 2 + self.num_coeffs]

        if len(features) < self.num_coeffs:
            features = np.pad(features, (0, self.num_coeffs - len(features)), 'constant')

        return features