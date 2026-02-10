from __future__ import annotations
import shutil
import sys
import zipfile
import tempfile
from pathlib import Path

ZIP_NAME = "Wild Animal Facing Extinction.zip"

def main_unpack(force_extract: bool = False):
    project_root = Path.cwd() 

    if project_root.name == "scripts": #zabezpieczenie na wypadek uruchomienia z katalogu scripts
         project_root = project_root.parent

    zip_path = project_root / ZIP_NAME
    dataset_dir = project_root / "dataset"

    #Jeśli dataset już jest (train/test/valid istnieją i nie są puste), to nie rozpakowuję już nic
    if not force_extract:
        ok = True
        for folder in ["train", "test", "valid"]:
            p = dataset_dir / folder
            if not p.exists() or not any(p.iterdir()):
                ok = False
                break

        if ok:
            print("Dataset już istnieje i nie jest pusty wiec pomijam rozpakowywanie")
            return

    if not zip_path.exists():
        print(f"Nie znaleziono pliku ZIP --> {zip_path}")
        sys.exit(2)

    #Temporary do rozpakowania głównego ZIP-a
    tmpdir = Path(tempfile.mkdtemp(prefix="unpack_dataset_"))
    try:
        print(f"Rozpakowuję główny ZIP --> {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmpdir)

        #W środku powinien być kolejny ZIP to biorę pierwszy znaleziony
        inner_zip = None
        for p in tmpdir.rglob("*.zip"):
            inner_zip = p
            break

        if inner_zip is None:
            print("Błąd --> nie znaleziono zagnieżdżonego ZIP-a w archiwum.")
            sys.exit(2)

        #Temp do rozpakowania zagnieżdżonego ZIP-a
        inner_tmpdir = Path(tempfile.mkdtemp(prefix="inner_unpack_"))
        try:
            print(f"Rozpakowuję zagnieżdżony ZIP: {inner_zip}")
            with zipfile.ZipFile(inner_zip, "r") as z:
                z.extractall(inner_tmpdir)

            dataset_dir.mkdir(parents=True, exist_ok=True)

            print("Przetwarzam rozpakowaną zawartość")
            moved = 0

            #Szukam "głównego folderu" (pierwszy katalog w inner_tmpdir)
            main_folder = None
            for item in inner_tmpdir.iterdir():
                if item.is_dir():
                    main_folder = item
                    break

            if main_folder is not None:
                print(f"Znaleziono główny folder: {main_folder.name}")

                #W środku szukam folderów --> rhino (usunąć), wild animals (przenieść podfoldery)
                for item in main_folder.iterdir():
                    if not item.is_dir():
                        continue

                    nazwa = item.name.lower()

                    if "rhino" in nazwa:
                        print(f"Usuwam folder: {item.name}/")
                        shutil.rmtree(item)

                    elif "wild animals" in nazwa:
                        print(f"Przetwarzam folder: {item.name}/")

                        #Przenoszę każdy folder zwierzaka do dataset/
                        for animal_folder in item.iterdir():
                            if animal_folder.is_dir() and not animal_folder.name.startswith("."):
                                target = dataset_dir / animal_folder.name

                                if target.exists():
                                    shutil.rmtree(target)

                                shutil.move(str(animal_folder), str(target))
                                moved += 1
                                print(f"--> przeniesiono: {animal_folder.name}/")

            #Jeśli w datasecie pojawiło się images/, to przenoszę train/test/valid poziom wyżej
            images_dir = dataset_dir / "images"
            if images_dir.exists():
                print("Wykryto folder images --> przenoszę train/test/valid do katalogu dataset/")

                for folder_name in ["train", "test", "valid"]:
                    source_folder = images_dir / folder_name
                    if source_folder.exists():
                        target_folder = dataset_dir / folder_name

                        if target_folder.exists():
                            shutil.rmtree(target_folder)

                        shutil.move(str(source_folder), str(target_folder))
                        moved += 1
                        print(f"--> przeniesiono {folder_name}/ do datasetu/")

                #Czyszczenie --> zostawiam tylko train/test/valid
                print("Sprzątam katalog dataset (zostawiam tylko train/test/valid)")
                for item in dataset_dir.iterdir():
                    if item.is_dir() and item.name not in ["train", "test", "valid"]:
                        print(f"Usuwam --> {item.name}/")
                        shutil.rmtree(item)

            print(f"Gotowe --> przeniesiono łącznie {moved} folderów.")

            wrapper_in_tmpdir = tmpdir / "Wild Animal Facing Extinction"
            if wrapper_in_tmpdir.exists():
                print(f"Usuwam opakowanie z folderu {wrapper_in_tmpdir}")
                shutil.rmtree(wrapper_in_tmpdir)

            print("Koniec")

        finally:
            shutil.rmtree(inner_tmpdir, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main_unpack()
