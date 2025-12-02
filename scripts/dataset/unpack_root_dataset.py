#%% Imports
from __future__ import annotations
import shutil
import sys
import zipfile
import tempfile
from pathlib import Path

#%% File name
ZIP_NAME = 'Wild Animal Facing Extinction.zip'

#%% Zip finder in root
def find_inner_zip(root: Path) -> Path | None:
    for p in root.rglob('*.zip'):
        return p
    return None

#%% Main unpacking function
def main_unpack(force_extract = False):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    zip_path = project_root / ZIP_NAME
    dataset_dir = project_root / 'dataset'
    
    if not force_extract and all((dataset_dir / folder).exists() and any((dataset_dir / folder).iterdir()) for folder in ['train', 'test', 'valid']):
        print("Dataset already exists and is not empty. Skipping unpacking.")
        return

    if not zip_path.exists():
        print(f'File not found: {zip_path}')
        sys.exit(2)

    tmpdir = Path(tempfile.mkdtemp(prefix='unpack_dataset_'))
    try:
        print(f'1) Unpacking main ZIP: {zip_path}')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        inner_zip = find_inner_zip(tmpdir)
        if not inner_zip:
            print('Error: no nested ZIP found inside the archive.')
            sys.exit(2)

        inner_tmpdir = Path(tempfile.mkdtemp(prefix='inner_unpack_'))
        try:
            print(f'2) Unpacking nested ZIP: {inner_zip}')
            with zipfile.ZipFile(inner_zip, 'r') as z:
                z.extractall(inner_tmpdir)

            dataset_dir = project_root / 'dataset'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            print(f'3) Processing extracted content')
            moved = 0
            
            # Find the main extracted folder
            main_folder = None
            for item in inner_tmpdir.iterdir():
                if item.is_dir():
                    main_folder = item
                    break
            
            if main_folder:
                print(f'   Found main folder: {main_folder.name}')
                
                # Process contents of the main folder
                for item in main_folder.iterdir():
                    if item.is_dir():
                        if 'rhino' in item.name.lower():
                            print(f'   Removing: {item.name}/')
                            shutil.rmtree(item)
                        elif 'wild animals' in item.name.lower():
                            print(f'   Processing: {item.name}/')
                            # Move contents of wild animals folder to dataset root
                            for animal_folder in item.iterdir():
                                if animal_folder.is_dir() and not animal_folder.name.startswith('.'):
                                    target = dataset_dir / animal_folder.name
                                    if target.exists():
                                        shutil.rmtree(target)
                                    shutil.move(str(animal_folder), str(target))
                                    moved += 1
                                    print(f'     → {animal_folder.name}/')

            # Additional processing: move train/test/valid folders from images to dataset root
            images_dir = dataset_dir / 'images'
            if images_dir.exists():
                print(f'4) Processing images directory')
                target_folders = ['train', 'test', 'valid']
                
                for folder_name in target_folders:
                    source_folder = images_dir / folder_name
                    if source_folder.exists():
                        target_folder = dataset_dir / folder_name
                        if target_folder.exists():
                            shutil.rmtree(target_folder)
                        shutil.move(str(source_folder), str(target_folder))
                        print(f'   → Moved {folder_name}/ to dataset root')
                        moved += 1
                
                # Remove remaining content in dataset directory except train/test/valid
                print(f'5) Cleaning up dataset directory')
                for item in dataset_dir.iterdir():
                    if item.is_dir() and item.name not in ['train', 'test', 'valid']:
                        print(f'   Removing: {item.name}/')
                        shutil.rmtree(item)

            print(f'Processed {moved} directories. Dataset structure cleaned up.')

            wrapper_in_tmpdir = tmpdir / 'Wild Animal Facing Extinction'
            if wrapper_in_tmpdir.exists():
                print(f'4) Removing wrapper folder: {wrapper_in_tmpdir}')
                shutil.rmtree(wrapper_in_tmpdir)

            print('\n✓ Done!')

        finally:
            shutil.rmtree(inner_tmpdir, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

#%% Run main unpack function
if __name__ == '__main__':
    main_unpack()
