import numpy as np
from skimage.feature import hog
from skimage.color import rgb2hsv


def policz_hog_kolor2_dla_obrazka(obraz_rgb,orientations=9,pixels_per_cell=(16, 16),cells_per_block=(2, 2),bins_hist=16):
    #normalizacja do [0, 1]
    if obraz_rgb.dtype != np.float32 and obraz_rgb.dtype != np.float64:
        obraz = obraz_rgb.astype("float32") / 255.0
    else:
        obraz = obraz_rgb

    #HOG dla każdego kanału RGB osobno!!
    hog_cechy = []
    for c in range(3):
        kanal = obraz[..., c]
        cechy_kanalu = hog(kanal,orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm="L2-Hys",feature_vector=True)
        hog_cechy.append(cechy_kanalu)
    hog_rgb = np.concatenate(hog_cechy, axis=0)

    #Histogram koloru na HSV
    hsv = rgb2hsv(obraz)
    hist_cechy = []
    for c in range(3):
        hist, _ = np.histogram(hsv[..., c],bins=bins_hist,range=(0.0, 1.0),density=True)
        hist_cechy.append(hist)

    hist_kolor = np.concatenate(hist_cechy, axis=0)  #3 * bins_hist

    #HOG + histogram koloru
    cechy_laczne = np.concatenate([hog_rgb, hist_kolor], axis=0)
    return cechy_laczne


def policz_hog_kolor2_batch(lista_obrazow,orientations=9,pixels_per_cell=(16, 16),cells_per_block=(2, 2),bins_hist=16):
    wszystkie_cechy = []

    for obraz in lista_obrazow:
        cechy = policz_hog_kolor2_dla_obrazka(obraz,orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,bins_hist=bins_hist)
        wszystkie_cechy.append(cechy)
    return np.array(wszystkie_cechy)
