import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Funktion: Baseline-Temperaturdaten mit zufälligen Anomalien erzeugen
def generate_baseline_temperature_with_random_anomalies(
        num_samples=10000,
        base_temp=22.0,
        noise_std=0.05,
        num_anomalies=70,
        anomaly_magnitude=5):
    np.random.seed(42)  # Für Reproduzierbarkeit

    # 1. Grundtemperatur mit sehr leichtem Rauschen
    temperature = base_temp + np.random.normal(0, noise_std, num_samples)

    # 2. Anomalie-Label initialisieren
    anomaly_labels = np.zeros(num_samples)

    # 3. Zufällige Indizes für Anomalien auswählen (ohne Wiederholung)
    anomaly_indices = np.random.choice(np.arange(num_samples), size=num_anomalies, replace=False)

    # 4. Anomalien einfügen (z. B. Temperaturspitzen)
    temperature[anomaly_indices] += anomaly_magnitude
    anomaly_labels[anomaly_indices] = 1.0

    # 5. Zeitstempel generieren
    timestamps = pd.date_range(start='2025-01-01', periods=num_samples, freq='min')

    # 6. DataFrame erstellen
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'anomaly_label': anomaly_labels
    })

    return df


# ==== Datensatz generieren ====
# Gleichmäßiger Verlauf mit 70 zufälligen, klaren Temperaturspitzen
baseline_df = generate_baseline_temperature_with_random_anomalies()

# ==== Exportieren ====
# Als CSV-Datei speichern
baseline_df.to_csv('baseline_temperature_with_70_anomalies.csv', index=False)

# ==== Plot erzeugen ====
plt.figure(figsize=(14, 6))

# 1. Temperaturverlauf zeichnen
plt.plot(baseline_df['timestamp'], baseline_df['temperature'], label='Temperaturverlauf', color='blue')

# 2. Anomalien mit roten "x" markieren
anomalies = baseline_df[baseline_df['anomaly_label'] == 1.0]
plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', marker='x', s=60, label='Anomalien')

# 3. Plot-Einstellungen
plt.xlabel('Zeitstempel')
plt.ylabel('Temperatur (°C)')
plt.title('Baseline-Temperaturverlauf mit 70 zufälligen Anomalien (Ausschläge nach oben)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 4. Plot speichern und anzeigen
plt.savefig('baseline_temperature_with_70_anomalies_plot.png')
plt.show()
