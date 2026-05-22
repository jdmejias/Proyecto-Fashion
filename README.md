# 🎓 Proyecto Final - Clasificación de Imágenes Fashion-MNIST con Machine Learning Clásico

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema completo de clasificación de imágenes** del dataset Fashion-MNIST, utilizando exclusivamente **Machine Learning clásico** y **extracción manual de características visuales**. El pipeline transforma imágenes 28×28 en escala de grises en un dataset tabular de características numéricas, sobre el cual se entrenan y comparan múltiples modelos de clasificación.

> ⚠️ **IMPORTANTE**: Este proyecto **NO** utiliza Deep Learning, Redes Neuronales Convolucionales (CNN), TensorFlow ni PyTorch para el entrenamiento de modelos. El enfoque es puramente de Machine Learning clásico con ingeniería de características.

---

## 🎯 Objetivo

Construir un sistema completo de clasificación de imágenes de principio a fin, que incluya:

1. **Carga y exploración** del dataset Fashion-MNIST
2. **Preprocesamiento digital** de imágenes (normalización, segmentación)
3. **Extracción manual de características** visuales (intensidad, forma, textura)
4. **Entrenamiento y comparación** de 9 modelos clásicos de ML
5. **Reducción de dimensionalidad** con PCA
6. **Validación cruzada** y búsqueda de hiperparámetros
7. **Evaluación exhaustiva** del mejor modelo
8. **Opción de rechazo** para predicciones inciertas

---

## 📊 Dataset

### Fashion-MNIST
- **Fuente**: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- **Total original**: 70,000 imágenes (60,000 train + 10,000 test)
- **Formato**: Imágenes 28×28 píxeles en escala de grises
- **Clases originales**: 10 categorías de prendas de vestir

### Clases Seleccionadas

Para este proyecto se seleccionaron **5 clases representativas** del total de 10:

| ID | Clase | Justificación |
|----|-------|--------------|
| 0 | T-shirt/top | Prenda básica, confundible con Shirt |
| 1 | Trouser | Silueta claramente diferente (vertical) |
| 3 | Dress | Silueta intermedia entre camiseta y pantalón |
| 6 | Shirt | Similar a T-shirt/top (caso difícil) |
| 8 | Bag | Categoría no-prenda, muy diferente |

**¿Por qué no usar todas las clases?**
- El proyecto se enfoca en demostrar el pipeline completo de ML clásico
- Un subconjunto representativo permite un análisis más profundo
- Incluye tanto clases fáciles (Bag vs Trouser) como difíciles (T-shirt vs Shirt)
- Reduce tiempos de entrenamiento sin sacrificar la complejidad del problema

---

## 📁 Estructura del Proyecto

```
fashion-mnist/
├── 📓 notebooks/
│   └── ProyectoFinal_FashionMNIST.ipynb    # Notebook principal
├── 📦 src/
│   ├── __init__.py                          # Inicialización del paquete
│   ├── data_loader.py                       # Carga y filtrado del dataset
│   ├── eda.py                               # Análisis Exploratorio de Datos
│   ├── preprocessing.py                     # Preprocesamiento de imágenes
│   ├── features.py                          # Extracción de características
│   ├── models.py                            # Modelos de ML y pipelines
│   ├── evaluation.py                        # Evaluación y métricas
│   ├── reject_option.py                     # Opción de rechazo
│   └── utils.py                             # Utilidades generales
├── 📊 outputs/
│   ├── figures/                             # Gráficas generadas
│   ├── tables/                              # Tablas comparativas (CSV)
│   └── models/                              # Modelos guardados (joblib)
├── 💾 data/
│   ├── fashion/                             # Dataset original (.gz)
│   └── processed/                           # Dataset tabular (CSV)
├── 📄 requirements.txt                      # Dependencias
└── 📄 README.md                             # Este archivo
```

---

## ⚙️ Requisitos e Instalación

### Requisitos previos
- Python 3.8+
- pip

### Instalación

```bash
# Clonar el repositorio (si aún no lo ha hecho)
git clone https://github.com/zalandoresearch/fashion-mnist.git
cd fashion-mnist

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales
- `numpy` - Computación numérica
- `pandas` - Manipulación de datos tabulares
- `matplotlib` / `seaborn` - Visualización
- `scikit-learn` - Modelos de ML, pipelines, métricas
- `scikit-image` - Procesamiento de imágenes (LBP, GLCM, HOG, Canny)
- `scipy` - Funciones científicas
- `joblib` - Serialización de modelos

---

## 🚀 Cómo Ejecutar

### Ejecución Local

```bash
# Opción 1: Jupyter Notebook
jupyter notebook notebooks/ProyectoFinal_FashionMNIST.ipynb

# Opción 2: JupyterLab
jupyter lab notebooks/ProyectoFinal_FashionMNIST.ipynb
```

### Ejecución en Google Colab

1. Subir el proyecto completo a Google Drive
2. Abrir el notebook en Colab
3. Montar Google Drive
4. Ejecutar las celdas en orden

El notebook detecta automáticamente si se ejecuta en Colab y ajusta las rutas.

---

## 🔄 Pipeline del Proyecto

1. **Carga del Dataset** → Leer archivos IDX comprimidos de Fashion-MNIST
2. **Filtrado de Clases** → Seleccionar las 5 clases representativas
3. **EDA Profundo** → Analizar distribuciones, brillo, contraste, texturas, bordes
4. **Preprocesamiento** → Normalizar, segmentar con Otsu, extraer ROI
5. **Extracción de Características** → 3 familias: intensidad, forma/bordes, textura
6. **Dataset Tabular** → Transformar imágenes a filas de características numéricas
7. **Entrenamiento** → 9 modelos clásicos con pipelines StandardScaler + Clasificador
8. **PCA** → Reducción de dimensionalidad y comparación
9. **Validación Cruzada** → StratifiedKFold (5 folds) + búsqueda de hiperparámetros
10. **Evaluación Final** → Matrices de confusión, métricas, análisis por clase
11. **Opción de Rechazo** → Rechazar predicciones con baja confianza

---

## 📈 Modelos Implementados

| # | Modelo | Tipo |
|---|--------|------|
| 1 | Perceptrón | Lineal |
| 2 | Adaline (SGDClassifier) | Lineal |
| 3 | Regresión Logística | Lineal |
| 4 | SVM Lineal | Margen |
| 5 | SVM Polinómico | Kernel |
| 6 | SVM RBF | Kernel |
| 7 | K-Nearest Neighbors | Basado en instancias |
| 8 | Árbol de Decisión | Basado en reglas |
| 9 | Bosque Aleatorio | Ensemble |

---

## 📊 Resultados Principales

Los resultados detallados se encuentran en el notebook y en los archivos:
- `outputs/tables/model_comparison.csv` - Comparación de todos los modelos
- `outputs/tables/pca_comparison.csv` - Comparación con/sin PCA
- `outputs/tables/reject_option_results.csv` - Resultados de opción de rechazo
- `outputs/figures/` - Todas las gráficas generadas

---

## ⚠️ Advertencia

Este proyecto utiliza exclusivamente **Machine Learning clásico** con extracción manual/ingenieril de características visuales. **NO** se emplean:

- ❌ Redes Neuronales Convolucionales (CNN)
- ❌ Deep Learning
- ❌ TensorFlow / PyTorch / Keras
- ❌ Transfer Learning
- ❌ Aprendizaje de representaciones automático

El objetivo pedagógico es demostrar que se puede construir un sistema de clasificación de imágenes competitivo usando técnicas clásicas de procesamiento de imágenes y Machine Learning.

---

## 👤 Autor

Proyecto Final - Aprendizaje de Máquina y Análisis de Datos

---

## 📄 Licencia

Este proyecto utiliza el dataset Fashion-MNIST bajo licencia MIT.
Código del proyecto: Uso educativo.
