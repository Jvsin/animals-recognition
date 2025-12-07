import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog


def policz_hog_batch(lista_obrazow,pixels_per_cell=(8, 8),cells_per_block=(2, 2),orientations=9):
    cechy = []

    for img in lista_obrazow:
        szary = rgb2gray(img)
        
        #Dalal & Triggs (2005)
        hog_vec = hog(szary,orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm="L2-Hys",visualize=False,feature_vector=True)
        cechy.append(hog_vec)

    return np.array(cechy)
