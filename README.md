# Vergleich von Half Space-Trees und inkrementellem Autoencoder

## Kurzbeschreibung
Dieses Projekt vergleicht zwei Methoden der Anomalieerkennung auf verschiedenen Datensätzen und speichert die Ergebnisse strukturiert ab.

## Voraussetzungen
- Python 3.x
- Siehe `requirements.txt` für alle benötigten Libraries.

## Nutzung
1. Passe die Pfade in der Konfigurationsdatei an:
    - `data_base_path`
    - `result_base_path`
2. Starte das Skript:
    ```
    python scripts/half_space_trees.py
    python scripts/incremental_auto_encoder.py
    ```
3. Optional: Passe den Schwellenwert (`threshold`) an, z.B. `threshold=3.1`.

## Datensätze
- baseline
- noise
- change
- drift
- extend

## Lizenz
MIT License

## Mitwirken
Beiträge sind willkommen! Bitte erstelle einen Pull Request.