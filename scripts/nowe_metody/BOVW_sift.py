import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans


def stworz_sift():
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create()
    return sift

#wyciagam deskryptory sift z jednego obrazu i zwracam tablice (N,128), jak brak punktow to (0,128)
def wyciagnij_sift_z_obrazu(image, sift, max_kp=200):
    if image.dtype != np.uint8:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image

    #RGB na gray
    if img_uint8.ndim == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8

    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)

    #ograniczam liczbe deskryptorów
    if descriptors.shape[0] > max_kp:
        idx = np.random.choice(descriptors.shape[0], size=max_kp, replace=False)
        descriptors = descriptors[idx]

    return descriptors.astype(np.float32)


def zbuduj_slownik_sift_bovw(obrazy_train,n_clusters=256,max_images=1000,max_kp_per_image=200,random_state=42):
    """
    Buduje słownik (codebook) BoVW:
    - bierze max_images obrazów z train,
    - z każdego wyciąga max_kp_per_image deskryptorów,
    - skleja wszystko i robi MiniBatchKMeans(n_clusters).

    Zwraca:
    - sift: obiekt SIFT
    - kmeans: wytrenowany MiniBatchKMeans
    """
    sift = stworz_sift()
    rng = np.random.RandomState(random_state)

    n = len(obrazy_train)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    idxs = idxs[: min(max_images, n)]

    wszystkie_desc = []

    print("Zbieram deskryptory SIFT do budowy słownika")
    for i in idxs:
        desc = wyciagnij_sift_z_obrazu(obrazy_train[i], sift, max_kp=max_kp_per_image)
        if desc.shape[0] > 0:
            wszystkie_desc.append(desc)

    if len(wszystkie_desc) == 0:
        raise ValueError("Nie udało się wyciągnąć żadnych deskryptorów SIFT.")

    wszystkie_desc = np.vstack(wszystkie_desc)
    print("Łączna liczba deskryptorów do KMeans:", wszystkie_desc.shape[0])
    print("trening MiniBatchKMeans (słownik BoVW)...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,random_state=random_state,batch_size=1000,verbose=1,)
    kmeans.fit(wszystkie_desc)

    return sift, kmeans


def policz_bovw_dla_obrazow(obrazy, sift, kmeans, max_kp_per_image=200):
    """
    Zamienia listę obrazów na macierz BoVW:
    - każdy obraz --> SIFT
    - SIFT --> przypisanie do najbliższego centroidu (słowo wizualne)
    - słowa --> histogram (z-normalizowany do sumy 1)
    Zwraca: X o kształcie (n_obrazów, n_clusters)
    """
    n = len(obrazy)
    k = kmeans.n_clusters
    X = np.zeros((n, k), dtype=np.float32)

    print("Liczenie histogramów BoVW dla obrazów...")
    for i, img in enumerate(obrazy):
        desc = wyciagnij_sift_z_obrazu(img, sift, max_kp=max_kp_per_image)
        if desc.shape[0] == 0:
            continue

        labels = kmeans.predict(desc)
        hist, _ = np.histogram(labels, bins=np.arange(k + 1))

        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s

        X[i] = hist

        if (i + 1) % 500 == 0:
            print(f"Przetworzono {i+1}/{n} obrazów...")

    return X
