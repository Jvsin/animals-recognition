import cv2
import numpy as np
import os
import glob

class TextureMatcher:
    def __init__(self, textures_dir, class_names, base_size=80):
        self.textures_dir = textures_dir
        self.class_names = class_names
        self.base_size = base_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #CLAHE --> algorytm, który "wyciąga" wzory (pasy, cętki) z tła
        self.templates_map = self._load_templates()

    def _preprocess(self, img):
        """Skaluje do bazy i poprawia kontrast (CLAHE)"""
        #Skalowanie do base_size (zachowując proporcje)
        h, w = img.shape[:2]
        if h > self.base_size or w > self.base_size:
            scale = self.base_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        #wyrównanie oświetlenia (kluczowe dla tekstur!)
        return self.clahe.apply(img)

    def _load_templates(self):
        templates = {}
        print(f"Wczytywanie tekstur z: {self.textures_dir} (Multi-Scale + CLAHE)...")
        
        for cls in self.class_names:
            path = os.path.join(self.textures_dir, cls)
            if not os.path.exists(path):
                templates[cls] = []
                continue

            imgs = []
            extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path, ext)))
                files.extend(glob.glob(os.path.join(path, ext.upper())))

            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_prep = self._preprocess(img) #Preprocessing wzorca
                    imgs.append(img_prep)
            
            templates[cls] = imgs
            print(f"  Klasa '{cls}': wczytano {len(imgs)} unikalnych wzorców.")
        
        return templates

    def compute_features(self, images_rgb):
        n_images = len(images_rgb)
        n_classes = len(self.class_names)
        features = np.zeros((n_images, n_classes), dtype=np.float32)

        #Definiujemy skale - sprawdzamy wzorzec w 3 rozmiarach --> 0.8x (jeśli zwierzę jest dalej), 1.0x (normalnie), 1.2x (jeśli bliżej)
        scales = [0.75, 1.0, 1.25]

        print("Liczenie tekstur (Multi-Scale Template Matching)...")
        
        for i, img_rgb in enumerate(images_rgb):
            #Konwersja i preprocessing badanego obrazu
            if img_rgb.dtype != np.uint8:
                img_gray_raw = (img_rgb * 255).astype(np.uint8)
            else:
                img_gray_raw = img_rgb
            
            if img_gray_raw.ndim == 3:
                img_gray_raw = cv2.cvtColor(img_gray_raw, cv2.COLOR_RGB2GRAY)
            
            #CLAHE też na badanym obrazie, żeby "wyciągnąć" tekstury
            img_gray = self.clahe.apply(img_gray_raw)
            img_h, img_w = img_gray.shape

            for c_idx, cls in enumerate(self.class_names):
                templates = self.templates_map[cls]
                if not templates:
                    features[i, c_idx] = 0.0
                    continue

                best_score_class = 0.0
                
                for base_tmpl in templates:
                    #Sprawdzam ten sam wzorzec w kilku skalach
                    for scale in scales:
                        #Skalowanie wzorca
                        t_h, t_w = base_tmpl.shape
                        new_t_w, new_t_h = int(t_w * scale), int(t_h * scale)
                        
                        #Jeśli po przeskalowaniu wzorzec jest za duży, pomiń
                        if new_t_h >= img_h or new_t_w >= img_w:
                            continue
                            
                        #Jeśli jest za malutki (szum), też pomiń
                        if new_t_h < 10 or new_t_w < 10:
                            continue

                        resized_tmpl = cv2.resize(base_tmpl, (new_t_w, new_t_h))

                        #Dopasowanie
                        res = cv2.matchTemplate(img_gray, resized_tmpl, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(res)
                        
                        if max_val > best_score_class:
                            best_score_class = max_val

                features[i, c_idx] = best_score_class

            if (i + 1) % 500 == 0:
                print(f"  Tekstury: {i+1}/{n_images}...")

        return features