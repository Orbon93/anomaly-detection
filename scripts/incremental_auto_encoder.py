import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import psutil
import json
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt


# Zufalls-Seed setzen für Reproduzierbarkeit
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Funktion zur Erstellung eines einfachen Autoencoders
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return autoencoder

# Basispfade und Parameter definieren
data_base_path = "../data/drift/"
result_base_path = "../results/drift/"
data_source = data_base_path + "dataset.csv"
anomaly_file = result_base_path + "iae_results/anomalien_online_autoencoder.txt"
metrics_file = result_base_path + "iae_results/metrics_online_autoencoder.json"

# Parameter für Training, Puffer und Anomalieerkennung
error_memory = []
anomaly_timestamps_list = []
error_values_list = []
error_memory_size = 2000
initial_training_window_size = 1500
buffer_limit = 2000
old_data_fraction = 0.1
threshold = 1.3

# Initialisierung von Modell, Puffer und Flags
autoencoder = None
buffer = []
training_done = False

# Initialisierung von Metriken zur Leistungsüberwachung
metrics = {
    "time_elapsed": 0.0,
    "messages_processed": 0,
    "throughput": 0.0,
    "latency": 0.0,
    "cpu_usage": 0.0,
    "ram_usage": 0.0
}
max_cpu_usage = 0.0
max_ram_usage = 0.0

# Lese erste Trainingsdaten zur Vorbereitung
initial_train_data = pd.read_csv(data_source, usecols=['value'], nrows=initial_training_window_size)

# Startzeit und Prozessobjekt für die Ressourcenmessung
start_time = time.time()
process = psutil.Process()
process.cpu_percent(None)

# Hauptverarbeitungsschleife: Zeilenweise Verarbeitung des Streams
for index, chunk in enumerate(pd.read_csv(data_source, iterator=True, chunksize=1)):
    chunk = chunk[['timestamp', 'value']]
    new_value = chunk['value'].values[0]

    # Modell bei Bedarf initialisieren
    if autoencoder is None:
        autoencoder = build_autoencoder(input_dim=chunk[['value']].shape[1])

    # Aktuellen Wert zum Puffer hinzufügen
    buffer.append(new_value)
    if len(buffer) > buffer_limit:
        buffer.pop(0)

    # Initiales Training (online auf Batch-Größe)
    if not training_done and len(buffer) < initial_training_window_size:
        X = np.array(buffer).reshape(-1, 1)
        autoencoder.train_on_batch(X, X)

    # Setze Flag, wenn Trainingsdaten vollständig vorliegen
    if len(buffer) >= initial_training_window_size and not training_done:
        training_done = True

    # Anomalieerkennung nach Abschluss des Initialtrainings
    if training_done:
        # Nach jeder zehnten Zeile: Online-Training mit neueren + alten Daten
        if index % 10 == 0:
            old_data_count = int(old_data_fraction * len(buffer))
            training_data = np.vstack([
                np.array(buffer[-initial_training_window_size:]).reshape(-1, 1),
                np.array(buffer[:old_data_count]).reshape(-1, 1)
            ])
            autoencoder.train_on_batch(training_data, training_data)

        # Vorhersage für aktuellen Datenpunkt und Berechnung des Rekonstruktionsfehlers
        new_X = chunk[['value']].values.reshape(-1, 1)
        reconstructed = autoencoder.predict(new_X, verbose=0)
        reconstruction_error = np.abs(reconstructed - new_X)

        # Fehlerwert speichern
        error_memory.append(reconstruction_error[0][0])
        error_values_list.append(reconstruction_error[0][0])
        if len(error_memory) > error_memory_size:
            error_memory.pop(0)

        # Dynamische Schwellenwertbestimmung auf Basis des Fehlerpuffers
        if len(error_memory) >= 50:
            mean_error = np.mean(error_memory)
            std_error = np.std(error_memory)
            threshold_value = mean_error + threshold * std_error

        # Anomalie erkannt, wenn aktueller Fehler den Schwellenwert übersteigt
        if len(error_memory) >= 50 and reconstruction_error[0][0] > threshold_value:
            anomaly_timestamps = chunk['timestamp'].iloc[0]
            anomaly_timestamps_list.append(anomaly_timestamps)
            with open(anomaly_file, 'a') as f:
                f.write(f"{anomaly_timestamps}\n")

    # Zeit- und Systemmetriken messen
    current_time = time.time()
    time_elapsed = current_time - start_time
    messages_processed = index + 1
    throughput = messages_processed / time_elapsed if time_elapsed > 0 else 0
    latency = time_elapsed / messages_processed if messages_processed > 0 else 0

    # CPU- und RAM-Nutzung des Prozesses messen
    process_cpu_usage = process.cpu_percent(interval=None)
    process_memory_usage = process.memory_info().rss / (1024 * 1024)

    # Maximalwerte aktualisieren
    max_cpu_usage = max(max_cpu_usage, process_cpu_usage)
    max_ram_usage = max(max_ram_usage, process_memory_usage)

    # Metriken aktualisieren
    metrics["time_elapsed"] = time_elapsed
    metrics["messages_processed"] = messages_processed
    metrics["throughput"] = throughput
    metrics["latency"] = latency
    metrics["cpu_usage"] = max_cpu_usage
    metrics["ram_usage"] = max_ram_usage

    # Minimale Wartezeit simulieren
    time.sleep(0.00001)

    # Metriken in JSON-Datei schreiben
    if index % 10 == 0:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    # Ausgabe zur Kontrolle: Alle 10 Durchläufe Anzahl verarbeiteter Werte
    if index % 10 == 0:
        print(f"Verarbeitete Werte: {index}")


# Konvertiere 'timestamp' in datetime-Format
df = pd.read_csv(data_source)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fehlerwerte mit Zeitstempeln speichern
error_scores_output_path = result_base_path + "iae_results/error_scores_autoencoder.txt"
with open(error_scores_output_path, "w") as f:
    for ts, err in zip(df['timestamp'][:len(error_values_list)], error_values_list):
        f.write(f"{ts},{err}\n")

# Anomalie-Timestamps laden
anomaly_timestamps_list = pd.to_datetime(anomaly_timestamps_list)

# Anomalien im DataFrame markieren
anomalies_df = df[df['timestamp'].isin(anomaly_timestamps_list)]

# Diagramm: Temperaturverlauf mit Anomalien
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['value'], label='Temperaturverlauf', color='blue')

if not anomalies_df.empty:
    plt.scatter(anomalies_df['timestamp'], anomalies_df['value'], color='red', label='Anomalien', marker='x', s=100, zorder=2)

if len(df) > 1500:
    timestamp_1500 = df.iloc[1500]['timestamp']
    plt.axvline(x=timestamp_1500, color='black', linestyle='dashed', linewidth=2, label='Index 1500')

plt.xlabel("Zeitstempel")
plt.ylabel("Temperatur")
plt.title("Temperaturverlauf mit erkannten Anomalien (Autoencoder)")
plt.legend()
plt.grid()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(result_base_path + "iae_results/autoencoder_anomalien_plot.png")

# Diagramm: Verlauf der Rekonstruktionsfehler
error_scores_df = pd.read_csv(
    error_scores_output_path,
    names=['timestamp', 'score'],
    sep=','
)
error_scores_df['timestamp'] = pd.to_datetime(error_scores_df['timestamp'])
error_scores_df['score'] = pd.to_numeric(error_scores_df['score'])

plt.figure(figsize=(12, 5))
plt.plot(error_scores_df['timestamp'], error_scores_df['score'], color='gold', label='Rekonstruktionsfehler')
plt.xlabel("Zeitstempel")
plt.ylabel("Fehlerscore")
plt.title("Verlauf der Anomalie-Scores (Autoencoder)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(result_base_path + "iae_results/autoencoder_error_scores_plot.png")

# Pfad zur Datei mit echten Anomalien (Ground Truth)
true_anomalies_txt_path = data_base_path + 'trimmed_anomalous_timestamps_csv.txt'

# Funktion zum Einlesen der Timestamps
def read_timestamps_from_txt(txt_file_path):
    with open(txt_file_path, 'r') as file:
        timestamps = file.readlines()
    timestamps = [timestamp.strip() for timestamp in timestamps if timestamp.strip() and timestamp.strip() != "timestamp"]
    return pd.to_datetime(timestamps, errors='coerce').dropna()

# Lade die tatsächlichen Anomalien und erkannte IAE-Anomalien
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
print("\n[Bewertung der IAE-Erkennung]")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
print(f"Accuracy:  {accuracy:.2f}")

# Optional: Speichere die Metriken in Datei
evaluation_output_path = result_base_path + "iae_results/iae_evaluation_metrics.json"
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
