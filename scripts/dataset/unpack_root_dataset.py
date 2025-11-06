from __future__ import annotations

import shutil
import sys
import zipfile
import tempfile
from pathlib import Path

ZIP_NAME = 'Dataset Of Animal Images.zip'


def find_inner_zip(root: Path) -> Path | None:
    for p in root.rglob('*.zip'):
        return p
    return None

def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    zip_path = project_root / ZIP_NAME

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

            print(f'3) Moving species folders to {dataset_dir}')
            moved = 0
            
            wrapper_folder = None
            for item in inner_tmpdir.iterdir():
                if item.is_dir() and 'dataset' in item.name.lower():
                    wrapper_folder = item
                    break
            
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

            print(f'Moved {moved} directories.')

            wrapper_in_tmpdir = tmpdir / 'Dataset Of Animal Images'
            if wrapper_in_tmpdir.exists():
                print(f'4) Removing wrapper folder: {wrapper_in_tmpdir}')
                shutil.rmtree(wrapper_in_tmpdir)

            print('\n✓ Done!')

        finally:
            shutil.rmtree(inner_tmpdir, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
