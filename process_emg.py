"""
Script de procesamiento de señales EMG
Realiza filtrado, transformada de Hilbert, análisis de conectividad y clasificación
"""

import numpy as np
import scipy.io
from scipy import signal
from scipy.stats import pearsonr
import json
import os
import time

def load_emg_data(filepath):
    """Carga los datos EMG desde archivo .mat"""
    data = scipy.io.loadmat(filepath)
    Fs = int(data['Fs'][0][0])  # Frecuencia de muestreo
    mSigM1 = data['mSigM1']  # Movimiento 1: (340000, 4)
    mSigM2 = data['mSigM2']  # Movimiento 2: (340000, 4)
    mSigM3 = data['mSigM3']  # Movimiento 3: (340000, 4)
    
    print(f"Frecuencia de muestreo: {Fs} Hz")
    print(f"Forma de señales: {mSigM1.shape}")
    
    return Fs, mSigM1, mSigM2, mSigM3

def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """Aplica filtro pasabanda Butterworth"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data, axis=0)
    return filtered

def compute_hilbert_transform(signal_data):
    """Calcula la transformada de Hilbert"""
    analytic_signal = signal.hilbert(signal_data, axis=0)
    envelope = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)
    return analytic_signal, envelope, phase

def segment_signal(signal_data, window_duration, fs):
    """Divide la señal en ventanas de duración especificada"""
    window_samples = int(window_duration * fs)
    n_samples, n_channels = signal_data.shape
    n_windows = n_samples // window_samples
    
    segments = []
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        segments.append(signal_data[start_idx:end_idx, :])
    
    return segments, n_windows

def compute_amplitude_correlation(envelope1, envelope2):
    """Calcula correlación de Pearson entre envolventes"""
    corr, _ = pearsonr(envelope1.flatten(), envelope2.flatten())
    return corr

def compute_phase_synchronization(phase1, phase2):
    """Calcula sincronización de fase usando Phase Locking Value (PLV)"""
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def compute_connectivity_matrices(envelopes, phases, n_channels=4):
    """Calcula matrices de conectividad para amplitud y fase"""
    n_events = len(envelopes)
    
    # Matrices de conectividad: amplitud y fase
    amp_connectivity = np.zeros((n_events, n_channels, n_channels))
    phase_connectivity = np.zeros((n_events, n_channels, n_channels))
    
    # Pares de canales: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for event_idx in range(n_events):
        # Diagonal es 1 (autocorrelación)
        for i in range(n_channels):
            amp_connectivity[event_idx, i, i] = 1.0
            phase_connectivity[event_idx, i, i] = 1.0
        
        # Calcular para cada par de canales
        for ch1, ch2 in channel_pairs:
            # Correlación de amplitud
            amp_corr = compute_amplitude_correlation(
                envelopes[event_idx][:, ch1],
                envelopes[event_idx][:, ch2]
            )
            amp_connectivity[event_idx, ch1, ch2] = amp_corr
            amp_connectivity[event_idx, ch2, ch1] = amp_corr  # Simétrica
            
            # Sincronización de fase
            phase_sync = compute_phase_synchronization(
                phases[event_idx][:, ch1],
                phases[event_idx][:, ch2]
            )
            phase_connectivity[event_idx, ch1, ch2] = phase_sync
            phase_connectivity[event_idx, ch2, ch1] = phase_sync  # Simétrica
    
    return amp_connectivity, phase_connectivity

def extract_upper_triangle_features(connectivity_matrix):
    """Extrae triángulo superior de matriz de conectividad"""
    n_events, n_channels, _ = connectivity_matrix.shape
    n_features = n_channels * (n_channels - 1) // 2  # 6 características
    
    features = np.zeros((n_events, n_features))
    
    for event_idx in range(n_events):
        # Extraer triángulo superior (sin diagonal)
        triu_indices = np.triu_indices(n_channels, k=1)
        features[event_idx, :] = connectivity_matrix[event_idx][triu_indices]
    
    return features

def process_movement(signal_data, Fs, movement_name, lowcut=100, highcut=200):
    """Procesa un movimiento completo"""
    print(f"\n=== Procesando {movement_name} ===")
    
    # 1. Filtrado pasabanda 100-200 Hz
    print("1. Aplicando filtro pasabanda 100-200 Hz...")
    filtered_signal = bandpass_filter(signal_data, lowcut, highcut, Fs)
    
    # 2-3. Transformada de Hilbert, envolvente y fase
    print("2-3. Calculando transformada de Hilbert, envolvente y fase...")
    analytic_signal, envelope, phase = compute_hilbert_transform(filtered_signal)
    
    # 4. Segmentación en ventanas de 10 segundos
    print("4. Segmentando en ventanas de 10 segundos...")
    window_duration = 10  # segundos
    filtered_segments, n_events = segment_signal(filtered_signal, window_duration, Fs)
    envelope_segments, _ = segment_signal(envelope, window_duration, Fs)
    phase_segments, _ = segment_signal(phase, window_duration, Fs)
    
    print(f"   Número de eventos: {n_events}")
    
    # 5. Análisis de conectividad
    print("5. Calculando conectividad entre canales...")
    amp_connectivity, phase_connectivity = compute_connectivity_matrices(
        envelope_segments, phase_segments
    )
    
    # 6. Matrices de conectividad ya están construidas (17×4×4)
    print("6. Matrices de conectividad construidas:", amp_connectivity.shape)
    
    # 7. Extracción de características
    print("7. Extrayendo características (triángulo superior)...")
    amp_features = extract_upper_triangle_features(amp_connectivity)
    phase_features = extract_upper_triangle_features(phase_connectivity)
    
    print(f"   Características de amplitud: {amp_features.shape}")
    print(f"   Características de fase: {phase_features.shape}")
    
    # Combinar características de amplitud y fase
    combined_features = np.hstack([amp_features, phase_features])  # 17 × 12
    
    results = {
        'movement_name': movement_name,
        'original_signal': signal_data.tolist(),
        'filtered_signal': filtered_signal.tolist(),
        'analytic_signal_real': np.real(analytic_signal).tolist(),
        'analytic_signal_imag': np.imag(analytic_signal).tolist(),
        'envelope': envelope.tolist(),
        'phase': phase.tolist(),
        'n_events': n_events,
        'filtered_segments': [seg.tolist() for seg in filtered_segments],
        'envelope_segments': [seg.tolist() for seg in envelope_segments],
        'phase_segments': [seg.tolist() for seg in phase_segments],
        'amp_connectivity': amp_connectivity.tolist(),
        'phase_connectivity': phase_connectivity.tolist(),
        'amp_features': amp_features.tolist(),
        'phase_features': phase_features.tolist(),
        'combined_features': combined_features.tolist(),
        'Fs': Fs,
        'window_duration': window_duration
    }
    
    return results

def main():
    """Función principal"""
    # Cargar datos
    filepath = 'sEMG_Mov123_4Chan.mat'
    print("Cargando datos EMG...")
    Fs, mSigM1, mSigM2, mSigM3 = load_emg_data(filepath)
    
    # Procesar cada movimiento
    results_m1 = process_movement(mSigM1, Fs, 'Movimiento 1')
    results_m2 = process_movement(mSigM2, Fs, 'Movimiento 2')
    results_m3 = process_movement(mSigM3, Fs, 'Movimiento 3')
    
    # Preparar datos para clasificación
    print("\n=== Preparando datos para clasificación ===")
    X_m1 = np.array(results_m1['combined_features'])  # 17 × 12
    X_m2 = np.array(results_m2['combined_features'])  # 17 × 12
    X_m3 = np.array(results_m3['combined_features'])  # 17 × 12
    
    # Crear matriz de características completa y etiquetas
    X_all = np.vstack([X_m1, X_m2, X_m3])  # 51 × 12
    y_all = np.hstack([
        np.zeros(X_m1.shape[0], dtype=int),  # Movimiento 1: clase 0
        np.ones(X_m2.shape[0], dtype=int),   # Movimiento 2: clase 1
        np.full(X_m3.shape[0], 2, dtype=int) # Movimiento 3: clase 2
    ])
    
    print(f"Matriz de características completa: {X_all.shape}")
    print(f"Etiquetas: {y_all.shape}")
    
    # Clasificación
    print("\n=== Realizando clasificación ===")
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    
    # Entrenar clasificador SVM con kernel RBF
    print(f"Entrenando SVM con {len(X_train)} muestras de entrenamiento...")
    start_time = time.time()
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Tiempo de entrenamiento: {training_time:.4f} segundos")
    
    # Predecir
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"Tiempo de predicción: {prediction_time:.4f} segundos")
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # PCA para visualización
    print("Calculando PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)
    
    # t-SNE para visualización
    print("Calculando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_all)-1))
    X_tsne = tsne.fit_transform(X_all)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Matriz de confusión:")
    print(cm)
    
    # Guardar todos los resultados
    all_results = {
        'Fs': Fs,
        'movement1': results_m1,
        'movement2': results_m2,
        'movement3': results_m3,
        'classification': {
            'X_all': X_all.tolist(),
            'y_all': y_all.tolist(),
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    }
    
    # Guardar en JSON (solo datos necesarios para visualización)
    # Reducir tamaño eliminando señales completas, solo mantener muestras para visualización
    print("\n=== Guardando resultados ===")
    
    # Para el dashboard, solo necesitamos muestras de las señales (no todas)
    sample_rate = 100  # Tomar 1 de cada 100 muestras para visualización
    
    dashboard_data = {
        'Fs': Fs,
        'sample_rate': sample_rate,
        'movements': {}
    }
    
    for mov_name, mov_data in [('movement1', results_m1), ('movement2', results_m2), ('movement3', results_m3)]:
        # Muestrear señales para visualización
        n_samples = len(mov_data['original_signal'])
        sample_indices = np.arange(0, n_samples, sample_rate)
        
        dashboard_data['movements'][mov_name] = {
            'name': mov_data['movement_name'],
            'original_signal': np.array(mov_data['original_signal'])[sample_indices].tolist(),
            'filtered_signal': np.array(mov_data['filtered_signal'])[sample_indices].tolist(),
            'analytic_signal_real': np.array(mov_data['analytic_signal_real'])[sample_indices].tolist(),
            'analytic_signal_imag': np.array(mov_data['analytic_signal_imag'])[sample_indices].tolist(),
            'envelope': np.array(mov_data['envelope'])[sample_indices].tolist(),
            'phase': np.array(mov_data['phase'])[sample_indices].tolist(),
            'n_events': mov_data['n_events'],
            'amp_connectivity': mov_data['amp_connectivity'],
            'phase_connectivity': mov_data['phase_connectivity'],
            'amp_features': mov_data['amp_features'],
            'phase_features': mov_data['phase_features'],
            'combined_features': mov_data['combined_features']
        }
    
    dashboard_data['classification'] = {
        'X_all': X_all.tolist(),
        'y_all': y_all.tolist(),
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'pca': {
            'X_pca': X_pca.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        },
        'tsne': {
            'X_tsne': X_tsne.tolist()
        }
    }
    
    # Guardar JSON para dashboard
    with open('data.json', 'w') as f:
        json.dump(dashboard_data, f)
    
    print("Resultados guardados en data.json")
    print("\n=== Procesamiento completado ===")

if __name__ == '__main__':
    main()

