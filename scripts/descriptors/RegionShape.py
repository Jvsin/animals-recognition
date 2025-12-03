import cv2
import numpy as np


class RegionShapeDescriptor:
    def __init__(self):
        # 7 Hu Moments (log-transformed)
        # 5 Geometric features  (Aspect Ratio, Extent, Solidity, Compactness, Eccentricity)
        self.vector_length = 7 + 5

    def describe(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(self.vector_length)

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area == 0:
            return np.zeros(self.vector_length)

        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()

        with np.errstate(divide='ignore', invalid='ignore'):
            hu = -1 * np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h

        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0

        if len(c) >= 5:
            (x_ell, y_ell), (MA, ma), angle = cv2.fitEllipse(c)
            a = MA / 2
            b = ma / 2
            if a > 0:
                if a < b:
                    a, b = b, a
                eccentricity = np.sqrt(1 - (b / a) ** 2)
            else:
                eccentricity = 0
        else:
            eccentricity = 0

        geometric_features = np.array([aspect_ratio, extent, solidity, compactness, eccentricity])

        full_vector = np.concatenate([hu, geometric_features])
        full_vector = np.nan_to_num(full_vector)

        return full_vector