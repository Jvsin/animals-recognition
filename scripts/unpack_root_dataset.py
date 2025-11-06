#!/usr/bin/env python3
"""
Rozpakuj główny ZIP, znajdź zagnieżdżony archiwum wewnątrz, rozpakuj go,
przenieś foldery gatunków do /dataset, usuń folder Dataset Of Animal Images.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
import tempfile
from pathlib import Path

ZIP_NAME = 'Dataset Of Animal Images.zip'


def find_inner_zip(root: Path) -> Path | None:
    """Znajdź pierwszy plik ZIP wewnątrz katalogu (zagnieżdżony archiwum)."""
    for p in root.rglob('*.zip'):
        return p
    return None


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    zip_path = project_root / ZIP_NAME

    if not zip_path.exists():
        print(f'Nie znaleziono pliku: {zip_path}')
        sys.exit(2)

    tmpdir = Path(tempfile.mkdtemp(prefix='unpack_dataset_'))
    try:
        # 1) Rozpakuj główny ZIP
        print(f'1) Rozpakowuję główny ZIP: {zip_path}')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        # 2) Znajdź zagnieżdżony ZIP wewnątrz
        inner_zip = find_inner_zip(tmpdir)
        if not inner_zip:
            print('Błąd: nie znaleziono zagnieżdżonego ZIP wewnątrz archiwum.')
            sys.exit(2)

        inner_tmpdir = Path(tempfile.mkdtemp(prefix='inner_unpack_'))
        try:
            print(f'2) Rozpakowuję zagnieżdżony ZIP: {inner_zip}')
            with zipfile.ZipFile(inner_zip, 'r') as z:
                z.extractall(inner_tmpdir)

            # 3) Przenieś foldery gatunków do /dataset
            dataset_dir = project_root / 'dataset'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            print(f'3) Przenoszę foldery gatunków do {dataset_dir}')
            moved = 0
            
            # Sprawdź czy jest folder "Dataset Of animal Images" w inner_tmpdir
            wrapper_folder = None
            for item in inner_tmpdir.iterdir():
                if item.is_dir() and 'dataset' in item.name.lower():
                    wrapper_folder = item
                    break
            
            # Jeśli znaleziony wrapper, przeszukaj jego zawartość
            if wrapper_folder:
                source_dir = wrapper_folder
            else:
                source_dir = inner_tmpdir
            
            for item in source_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    target = dataset_dir / item.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(item), str(target))
                    moved += 1
                    print(f'   → {item.name}/')

            print(f'Przeniesiono {moved} katalogów.')

            # 4) Usuń folder "Dataset Of Animal Images" z tmpdir jeśli istnieje
            wrapper_in_tmpdir = tmpdir / 'Dataset Of Animal Images'
            if wrapper_in_tmpdir.exists():
                print(f'4) Usuwam folder wrapper: {wrapper_in_tmpdir}')
                shutil.rmtree(wrapper_in_tmpdir)

            print('\n✓ Gotowe!')

        finally:
            shutil.rmtree(inner_tmpdir, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
