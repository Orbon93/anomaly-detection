# Benötigte Bibliotheken importieren
import pandas as pd
from river import anomaly, preprocessing
import numpy as np
import time
from collections import deque
import psutil
import json
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei mit den Temperaturdaten (inkl. künstlicher Anomalien)
data_base_path = "../data/baseline/"
result_base_path = "../results/baseline/"
csv_file_path = data_base_path + "dataset.csv"

# Einlesen der CSV-Datei als DataFrame
df = pd.read_csv(csv_file_path)

# Initialisierung von Listen für Anomalien, Scores und Fehlerwerte
anomalies = []                # Enthält erkannte Anomalien (Zeitstempel, Wert, Score)
scores = []                   # Enthält alle berechneten Scores
all_error_scores = []         # Enthält alle Fehlerwerte mit Zeitstempeln (für spätere Auswertung)
labels = []                   # Enthält Labels zur Evaluierung (1 = Anomalie, 0 = normal)

# Definition der Fenstergrößen und Schwellenwerte
window_size = 1000            # Fenstergröße für das HalfSpaceTrees-Modell
dynamic_window_size = 500     # Dynamisches Fenster für Schwellenwertanpassung
threshold = 5.5              # Multiplikator für dynamischen Schwellenwert

# Initialisierung des HalfSpaceTrees-Modells mit MinMax-Skalierung
hst = preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(
    n_trees=100,
    height=4,
    window_size=window_size,
    seed=42
)

# Name der Sensordaten-Spalte
sensor_column = 'value'

# Sliding Window für dynamische Schwellenwertberechnung
window_scores = deque(maxlen=dynamic_window_size)
dynamic_threshold = None

# Initialisierung von Metriken zur Leistungsüberwachung
metrics = {
    "time_elapsed": 0.0,
    "messages_processed": 0,
    "throughput": 0.0,
    "latency": 0.0,
    "cpu_usage": 0.0,
    "ram_usage": 0.0,
    "process_memory_usage": 0.0
}

# Variablen zur Überwachung der Ressourcennutzung
max_cpu_usage = 0.0
max_ram_usage = 0.0
metrics_file = result_base_path + "hst_results/metrics_dynamic_hst.json"

# Startzeit und Prozesserfassung für die Leistungsmessung
start_time = time.time()
process = psutil.Process()

# Initialisiere die CPU-Messung
process.cpu_percent(None)

# Hauptverarbeitungsschleife: Daten einlesen, lernen, Scoring, Anomalieerkennung
for idx, row in df.iterrows():
    feature_set = {sensor_column: row[sensor_column]}
    hst.learn_one(feature_set)
    score = hst.score_one(feature_set)
    scores.append(score)
    all_error_scores.append(f"{row['timestamp']},{score}")

    current_time = time.time()
    time_elapsed = current_time - start_time
    messages_processed = idx + 1
    throughput = messages_processed / time_elapsed if time_elapsed > 0 else 0
    latency = time_elapsed / messages_processed if messages_processed > 0 else 0

    # CPU und RAM realistisch messen
    process_cpu_usage = process.cpu_percent(interval=None)  # Nutzt 10ms Messintervall
    process_memory_usage = process.memory_info().rss / (1024 * 1024)

    max_cpu_usage = max(max_cpu_usage, process_cpu_usage)
    max_ram_usage = max(max_ram_usage, process_memory_usage)

    metrics["time_elapsed"] = time_elapsed
    metrics["messages_processed"] = messages_processed
    metrics["throughput"] = throughput
    metrics["latency"] = latency
    metrics["cpu_usage"] = max_cpu_usage
    metrics["ram_usage"] = max_ram_usage

    time.sleep(0.00001)

    if idx % 10 == 0:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    if idx >= window_size:
        window_scores.append(score)
        if len(window_scores) == dynamic_window_size:
            mean_score = np.mean(window_scores)
            std_dev_score = np.std(window_scores)
            dynamic_threshold = (mean_score + threshold * std_dev_score)

            if score >= dynamic_threshold:
                anomalies.append(f"{row['timestamp']},{row['value']},{score}")


# Speichern der detektierten Anomalien
anomaly_output_path = result_base_path + "hst_results/anomaly_timestamps_dynamic_hst.txt"
with open(anomaly_output_path, "w") as file:
    file.write("\n".join(anomalies))

# Speichern aller Fehlerwerte mit Zeitstempeln zur späteren Analyse
error_scores_output_path = result_base_path + "/hst_results/error_scores_dynamic_hst.txt"
with open(error_scores_output_path, "w") as file:
    file.write("\n".join(all_error_scores))

print(f"Anomalie-Zeitstempel gespeichert ({len(anomalies)} Anomalien erkannt).")
print(f"Fehlerscores gespeichert (insgesamt {len(all_error_scores)} Einträge).")


# Konvertiere 'timestamp' in datetime-Format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Anomalien aus Datei laden
with open(anomaly_output_path, 'r') as f:
    anomaly_timestamps_list = [line.strip().split(',')[0] for line in f.readlines()]  # Nur Timestamp extrahieren

# Konvertiere Anomalie-Timestamps in datetime
anomaly_timestamps_list = pd.to_datetime(anomaly_timestamps_list)

# Anomalien im DataFrame markieren
anomalies_df = df[df['timestamp'].isin(anomaly_timestamps_list)]

# Diagramm erzeugen
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'], label='Temperaturverlauf', color='blue')

if not anomalies_df.empty:
    plt.scatter(anomalies_df['timestamp'], anomalies_df['value'], color='red', label='Anomalien', marker='x', zorder=2, s=100)

if len(df) > 1500:
    timestamp_1500 = df.iloc[1500]['timestamp']
    plt.axvline(x=timestamp_1500, color='black', linestyle='dashed', linewidth=2, label='Index 1500')

plt.xlabel("Zeitstempel")
plt.ylabel("Temperatur")
plt.title("Temperaturverlauf mit erkannten Anomalien (HST)")
plt.legend()
plt.grid()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(result_base_path + "hst_results/hst_anomalien_plot.png")


# Fehlerwerte (Score) aus Datei laden
error_scores_df = pd.read_csv(
    error_scores_output_path,
    names=['timestamp', 'score'],
    sep=','
)

# Konvertiere Timestamps in datetime
error_scores_df['timestamp'] = pd.to_datetime(error_scores_df['timestamp'])
error_scores_df['score'] = pd.to_numeric(error_scores_df['score'])

# Diagramm für Fehlerscore-Verlauf erstellen
plt.figure(figsize=(12, 5))
plt.plot(error_scores_df['timestamp'], error_scores_df['score'], color='gold', label='Fehlerscore-Verlauf')
plt.xlabel("Zeitstempel")
plt.ylabel("Fehlerscore")
plt.title("Verlauf der Anomalie-Scores (HST)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(result_base_path + "hst_results/hst_error_scores_plot.png")


# Pfad zur Datei mit echten Anomalien (Ground Truth)
true_anomalies_txt_path = data_base_path + 'trimmed_anomalous_timestamps_csv.txt'

# Funktion zum Einlesen der Timestamps
def read_timestamps_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as file:
        timestamps = file.readlines()
    timestamps = [timestamp.strip() for timestamp in timestamps if timestamp.strip() and timestamp.strip() != "timestamp"]
    return pd.to_datetime(timestamps, errors='coerce').dropna()

# Lade die tatsächlichen Anomalien und erkannte HST-Anomalien
true_anomalies = read_timestamps_from_txt(true_anomalies_txt_path)
detected_anomalies = pd.to_datetime(anomaly_timestamps_list).dropna()

# Vergleichsfunktion mit Zeit-Toleranz (optional: z. B. ±60 Sekunden)
def get_labels(detected_anomalies, true_anomalies, tolerance=pd.Timedelta(seconds=60)):
    true_positive = 0
    false_positive = 0
    matched_true = set()

    for detected in detected_anomalies:
        match = False
        for true in true_anomalies:
            if abs(detected - true) <= tolerance and true not in matched_true:
                true_positive += 1
                matched_true.add(true)
                match = True
                break
        if not match:
            false_positive += 1

    false_negative = len(true_anomalies) - len(matched_true)
    return true_positive, false_positive, false_negative

# Evaluation: Berechnung der Metriken
def evaluate_model(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
    return precision, recall, f1, accuracy

# Metriken berechnen
tp, fp, fn = get_labels(detected_anomalies, true_anomalies)
precision, recall, f1, accuracy = evaluate_model(tp, fp, fn)

# Ausgabe der Ergebnisse
print("\n[Bewertung der HST-Erkennung]")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
print(f"Accuracy:  {accuracy:.2f}")

# Optional: Speichere die Metriken in Datei
evaluation_output_path = result_base_path + "hst_results/hst_evaluation_metrics.json"
with open(evaluation_output_path, 'w') as f:
    json.dump({
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy
    }, f, indent=4)