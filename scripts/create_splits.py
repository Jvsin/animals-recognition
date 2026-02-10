import os
import csv
import random
from pathlib import Path

random.seed(42)
final_classes = {'cheetah': 0, 'elephant': 1, 'giraffe': 2, 'lion': 3, 'rhino': 4, 'zebra': 5}

def create_splits(force_extract=False):
    """
    Tworzy pliki train.csv i test.csv.
    Obsługuje zarówno strukturę folderową (train/lion/img.jpg), 
    jak i płaską (train/lion_001.jpg).
    """
    
    script_path = Path(__file__).resolve()
    #Zabezpieczenie ścieżek
    if script_path.parent.name == 'scripts':
        project_root = script_path.parent.parent
    else:
        project_root = script_path.parent

    dataset_root = project_root / 'dataset'
    train_csv_path = dataset_root / 'train.csv'
    test_csv_path = dataset_root / 'test.csv'

    if not force_extract and train_csv_path.exists() and test_csv_path.exists():
        #Sprawdzamy czy pliki nie są puste (większe niż tylko nagłówek ~20 bajtów)
        if train_csv_path.stat().st_size > 50:
            print("Pliki CSV już istnieją i mają zawartość. Pomijam.")
            return

    print(f"Generowanie CSV z folderu: {dataset_root}")

    csv_files = {
        'train': open(train_csv_path, 'w', newline='', encoding='utf-8'),
        'test':  open(test_csv_path, 'w', newline='', encoding='utf-8')
    }
    writers = {k: csv.writer(v) for k, v in csv_files.items()}
    
    for w in writers.values():
        w.writerow(['image_path', 'class'])

    total_count = 0

    try:
        #Iterujemy po train, test, valid
        for split_dir in ['train', 'test', 'valid']:
            split_path = dataset_root / split_dir
            
            if not split_path.exists():
                continue

            #valid trafia do test.csv, reszta zgodnie z nazwą
            target_key = 'test' if split_dir == 'valid' else split_dir
            if target_key not in writers: target_key = 'test'
            writer = writers[target_key]

            print(f"Skanowanie folderu: {split_dir}...")

            #Pobieramy całą zawartość folderu
            items = list(split_path.iterdir())
            
            for item in items:
                #PRZYPADEK 1 --> Zdjęcia są w podfolderach (np. train/lion/...)
                if item.is_dir():
                    folder_name = item.name.lower()
                    matched_class = None
                    
                    #Sprawdzamy czy nazwa folderu zawiera nazwę klasy
                    for cls in final_classes:
                        if cls in folder_name:
                            matched_class = cls
                            break
                    
                    if matched_class:
                        for img in item.iterdir():
                            if img.is_file() and img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                rel_path = img.relative_to(dataset_root).as_posix()
                                writer.writerow([rel_path, matched_class])
                                total_count += 1

                #PRZYPADEK 2 --> Zdjęcia są luzem (np. train/lion_01.jpg)
                elif item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    file_name = item.name.lower()
                    matched_class = None
                    
                    #Sprawdzamy czy nazwa pliku zawiera nazwę klasy
                    for cls in final_classes:
                        if cls in file_name:
                            matched_class = cls
                            break
                    
                    if matched_class:
                        rel_path = item.relative_to(dataset_root).as_posix()
                        writer.writerow([rel_path, matched_class])
                        total_count += 1

    finally:
        for f in csv_files.values():
            f.close()

    print(f"Zakończono. Utworzono wpisy dla {total_count} obrazów.")
    if total_count == 0:
        print("UWAGA: Nadal 0 plików. Sprawdź, czy nazwy plików zawierają nazwy zwierząt (np. 'lion_1.jpg').")

if __name__ == "__main__":
    create_splits(force_extract=True)