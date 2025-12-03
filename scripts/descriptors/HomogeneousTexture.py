import cv2
import numpy as np


class HomogeneousTextureDescriptor:
    def __init__(self, num_rings=5, num_wedges=6):
        self.num_rings = num_rings
        self.num_wedges = num_wedges

    def describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        magnitude_spectrum = np.abs(fshift)

        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]

        rho = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        theta = np.rad2deg(theta)
        theta[theta < 0] += 360
        theta[theta >= 180] -= 180

        max_radius = min(h, w) // 2
        ring_step = max_radius / self.num_rings
        wedge_step = 180.0 / self.num_wedges

        global_mean = np.mean(magnitude_spectrum)
        global_std = np.std(magnitude_spectrum)

        features = [global_mean, global_std]

        for r in range(self.num_rings):
            r_inner = r * ring_step
            r_outer = (r + 1) * ring_step

            ring_mask = (rho >= r_inner) & (rho < r_outer)

            for w_idx in range(self.num_wedges):
                t_inner = w_idx * wedge_step
                t_outer = (w_idx + 1) * wedge_step

                wedge_mask = (theta >= t_inner) & (theta < t_outer)
                region_mask = ring_mask & wedge_mask
                region_pixels = magnitude_spectrum[region_mask]

                if len(region_pixels) > 0:
                    mean_val = np.mean(np.log(region_pixels + 1))
                    std_val = np.std(np.log(region_pixels + 1))
                else:
                    mean_val = 0.0
                    std_val = 0.0

                features.append(mean_val)
                features.append(std_val)

        return np.array(features)