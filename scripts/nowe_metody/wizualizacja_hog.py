import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import hog


def policz_hog_z_wizualizacja(img,pixels_per_cell=(8, 8),cells_per_block=(2, 2),orientations=9):
    szary = rgb2gray(img)
    hog_vec, hog_image = hog(szary,orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm="L2-Hys",visualize=True,feature_vector=True,)
    return hog_vec, hog_image


def pokaz_hog(img, idx=None,pixels_per_cell=(8, 8),cells_per_block=(2, 2),orientations=9):
    _, hog_img = policz_hog_z_wizualizacja(img,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,orientations=orientations)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Oryginał {idx}" if idx is not None else "Oryginał")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_img, cmap="gray")
    plt.axis("off")
    plt.title(f"HOG {idx}" if idx is not None else "HOG")

    plt.tight_layout()
    plt.show()
