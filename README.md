# Análisis de Señales Musculares (EMG) para Clasificar y Reconocer Diferentes Movimientos

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Sistema completo de procesamiento y análisis de señales electromiográficas (EMG) de superficie para la clasificación de movimientos mediante análisis de conectividad funcional.

## 🌐 Dashboard en Vivo

**👉 [Ver Dashboard Interactivo](https://felirangelp.github.io/Analisis-de-Senales-Musculares-EMG-/)**

> **Nota**: El dashboard requiere datos procesados. Para ver el dashboard completo con datos, sigue las instrucciones de instalación local más abajo.

## 📋 Descripción

Este proyecto implementa un pipeline completo de procesamiento de señales EMG que incluye:

- **Preprocesamiento**: Filtrado pasabanda y transformada de Hilbert
- **Análisis de Conectividad**: Correlación de amplitud y sincronización de fase entre canales
- **Extracción de Características**: Vectores de características basados en matrices de conectividad
- **Clasificación**: Reconocimiento de movimientos usando Machine Learning
- **Visualización Interactiva**: Dashboard web con visualizaciones interactivas

## 🎯 Objetivo

Desarrollar un sistema que pueda identificar y clasificar diferentes tipos de movimientos corporales analizando las señales eléctricas generadas por los músculos (EMG). El sistema utiliza técnicas de procesamiento de señales y machine learning para extraer patrones característicos de cada movimiento.

## 🚀 Características Principales

### 1. Preprocesamiento de Señales
- Filtrado pasabanda (100-200 Hz) para eliminar ruido y artefactos
- Transformada de Hilbert para obtener señal analítica
- Cálculo de envolvente (amplitud instantánea) y fase instantánea

### 2. Análisis de Conectividad
- **Correlación de Amplitud**: Mide la similitud en la modulación de amplitud entre canales
- **Sincronización de Fase (PLV)**: Cuantifica el grado de sincronización de fase entre canales
- Matrices de conectividad 4×4 para cada ventana temporal

### 3. Extracción de Características
- Extracción del triángulo superior de matrices de conectividad
- 6 características de amplitud + 6 características de fase = 12 características por evento
- Segmentación temporal en ventanas de 10 segundos

### 4. Clasificación
- Algoritmo: Support Vector Machine (SVM) con kernel RBF
- Visualización: PCA y t-SNE para reducción de dimensionalidad
- Métricas: Accuracy, matriz de confusión, precision, recall, F1-score

### 5. Dashboard Interactivo
- Visualizaciones interactivas con Plotly.js
- Navegación por pestañas (Movimiento 1, 2, 3, Clasificación)
- Análisis y conclusiones para cada sección

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- Archivo de datos EMG en formato `.mat` (estructura: `Fs`, `mSigM1`, `mSigM2`, `mSigM3`)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/felirangelp/Analisis-de-Senales-Musculares-EMG-.git
cd Analisis-de-Senales-Musculares-EMG-
```

2. **Crear ambiente virtual**
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

## 💻 Uso

### 1. Preparar los Datos

Coloca tu archivo de datos EMG (`.mat`) en el directorio del proyecto. El archivo debe contener:
- `Fs`: Frecuencia de muestreo (Hz)
- `mSigM1`: Señal del Movimiento 1 (array: N muestras × 4 canales)
- `mSigM2`: Señal del Movimiento 2 (array: N muestras × 4 canales)
- `mSigM3`: Señal del Movimiento 3 (array: N muestras × 4 canales)

### 2. Procesar los Datos

Ejecuta el script de procesamiento:

```bash
python3 process_emg.py
```

Este script realizará:
- ✅ Carga de datos desde el archivo `.mat`
- ✅ Filtrado pasabanda 100-200 Hz
- ✅ Transformada de Hilbert
- ✅ Cálculo de envolvente y fase
- ✅ Segmentación en ventanas de 10 segundos
- ✅ Análisis de conectividad entre canales
- ✅ Extracción de características
- ✅ Clasificación de movimientos
- ✅ Generación de `data.json` para el dashboard

### 3. Visualizar Resultados

Inicia el servidor del dashboard:

```bash
python3 server.py
```

El dashboard estará disponible en: **http://localhost:8013/dashboard_v2.html**

El navegador se abrirá automáticamente. Si no, abre manualmente la URL.

## 📁 Estructura del Proyecto

```
.
├── process_emg.py          # Script principal de procesamiento
├── dashboard_v2.html       # Dashboard interactivo con pestañas
├── index.html              # Dashboard para GitHub Pages
├── server.py              # Servidor HTTP para el dashboard
├── requirements.txt       # Dependencias Python
├── .gitignore            # Archivos excluidos del repositorio
└── README.md             # Este archivo
```

## 🔬 Metodología

### Pipeline de Procesamiento

```
Señal EMG Original
    ↓
Filtrado Pasabanda (100-200 Hz)
    ↓
Transformada de Hilbert
    ↓
Envolvente + Fase Instantánea
    ↓
Segmentación (ventanas de 10s)
    ↓
Análisis de Conectividad
    ↓
Extracción de Características
    ↓
Clasificación (SVM)
```

### Métricas de Conectividad

1. **Correlación de Amplitud**: Correlación de Pearson entre envolventes de pares de canales
2. **Phase Locking Value (PLV)**: Sincronización de fase entre canales
   ```
   PLV = |mean(exp(i(φ₁ - φ₂)))|
   ```

### Características Extraídas

Para cada evento, se extraen 12 características:
- 6 características de correlación de amplitud (pares: 1-2, 1-3, 1-4, 2-3, 2-4, 3-4)
- 6 características de sincronización de fase (mismos pares)

## 📊 Resultados Esperados

El sistema genera:
- **Matrices de conectividad**: 17 eventos × 4 canales × 4 canales
- **Matriz de características**: 51 eventos (17×3) × 12 características
- **Accuracy de clasificación**: Típicamente >90% con datos bien balanceados

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje principal
- **NumPy & SciPy**: Procesamiento de señales
- **scikit-learn**: Machine Learning (SVM, PCA, t-SNE)
- **Plotly.js**: Visualizaciones interactivas
- **HTML/CSS/JavaScript**: Dashboard web

## 📚 Referencias y Conceptos Clave

### Señales EMG
Las señales electromiográficas (EMG) registran la actividad eléctrica de los músculos. Son útiles para:
- Control de prótesis
- Rehabilitación
- Análisis de movimiento
- Interfaces humano-computadora

### Transformada de Hilbert
Permite obtener la señal analítica compleja, de la cual se extraen:
- **Envolvente**: Amplitud modulada de la señal
- **Fase**: Información temporal y de sincronización

### Conectividad Funcional
Mide las relaciones entre diferentes canales/canales, indicando:
- Coordinación muscular
- Sincronización temporal
- Patrones de activación

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Felipe Rangel Perez**

- Procesamiento de Señales Biológicas
- Pontificia Universidad Javeriana

## 🙏 Agradecimientos

- Pontificia Universidad Javeriana
- Comunidad de procesamiento de señales biológicas

## 📧 Contacto

Para preguntas o sugerencias, puedes abrir un issue en el repositorio.

---

⭐ Si este proyecto te resultó útil, considera darle una estrella en GitHub!
