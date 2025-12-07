import numpy as np
import cv2


def _prepare_rgb_uint8(image):
    img = image
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _fill_holes(mask_bool):
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    h, w = mask_u8.shape
    im_floodfill = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled = mask_u8 | im_floodfill_inv
    return filled > 0


def draw_orb_keypoints(image, nfeatures=1000):
    img_u8 = _prepare_rgb_uint8(image)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps = orb.detect(gray, None)
    out = img_u8.copy()
    out = cv2.drawKeypoints(img_u8,kps,out,color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,)
    out = out.astype(np.float32) / 255.0
    return out


def remove_background_from_image(image,nfeatures=1500,margin_frac=0.05,bg_color=(1.0, 1.0, 1.0),):
    """
    Usuwanie tła na podstawie lokalizacji punktów ORB:
    --> wykrywamy punkty ORB,
    --> bierzemy ich convex hull (zewnętrzna otoczka),
    --> wypełniam cały środek tej otoczki,
    --> lekko rozszerzamy maskę (margines),
    --> resztę ustawiamy na kolor tła.
    """
    img_u8 = _prepare_rgb_uint8(image)
    h, w = img_u8.shape[:2]

    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps = orb.detect(gray, None)

    #ORB nie znalazł nic to zostawiam środek obrazka
    if len(kps) == 0:
        mask = np.zeros((h, w), dtype=bool)
        y1 = int(0.2 * h)
        y2 = int(0.8 * h)
        x1 = int(0.2 * w)
        x2 = int(0.8 * w)
        mask[y1:y2, x1:x2] = True
    else:
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        hull = cv2.convexHull(pts)
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, hull.astype(np.int32), 255)

        #margines
        margin = int(margin_frac * max(h, w))
        if margin > 0:
            kernel = np.ones((margin, margin), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

        mask = mask_u8 > 0
        mask = _fill_holes(mask)

        #lekkie domknięcie (wygładza brzegi)
        kernel_small = np.ones((5, 5), np.uint8)
        mask_u8 = (mask.astype(np.uint8)) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        mask = mask_u8 > 0

    #maska na obraz
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0

    out = img.copy()
    bg = np.array(bg_color, dtype=np.float32)
    out[~mask] = bg

    return out
