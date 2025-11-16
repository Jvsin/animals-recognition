# animals-recognition

Instrukcje dotyczące pobrania i rozpakowania datasetu Mendeley oraz przygotowania środowiska.

Pliki dodane w tym repozytorium:
- `requirements.txt` — wymagane pakiety do pobrania i rozpakowania archiwów (requests, tqdm, py7zr, beautifulsoup4)
- `scripts/unpack_root_dataset.py` — prosty skrypt, który szuka w root projektu pliku `Wild Animal Facing Extinction.zip`, rozpakowuje go i tworzy katalogi `dataset/<species>/` (używa pliku etykiet jeśli dostępny lub próbuje inferencji z nazw plików)

Jak użyć
1) Zainstaluj zależności (zalecane w wirtualnym środowisku):

```powershell
python -m pip install -r requirements.txt
```

2) Ręcznie pobierz plik ze strony Mendeley (https://data.mendeley.com/datasets/vhmvfbgvxj/2) i umieść go w katalogu głównym repozytorium pod nazwą:

```
Wild Animal Facing Extinction.zip
```

3) Uruchom skrypt, który automatycznie rozpakowuje i organizuje pliki:

```powershell
python .\scripts\unpack_root_dataset.py
```

Skrypt rozpakowuje archiwum do katalogu `dataset/`. Jeśli w archiwum znajduje się plik etykiet (np. `labels.csv` lub `annotations.csv`) zawierający kolumny `filename,label`, skrypt spróbuje przenieść pliki do `dataset/<label>/`.

Uwaga
# Niektóre archiwa (np. `.rar`) mogą wymagać dodatkowych narzędzi i nie są obsługiwane domyślnie.
# animals-recognition
analysis of digital images project

