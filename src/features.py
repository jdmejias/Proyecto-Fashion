# -*- coding: utf-8 -*-
"""
Módulo de extracción de características visuales - Proyecto Fashion-MNIST.

Este es el CORAZÓN del proyecto. Transforma cada imagen 28x28 en un vector
de características numéricas usando tres familias:
1. Características de Intensidad
2. Características de Forma y Bordes
3. Características de Textura

NO se usa deep learning. Todas las características se extraen manualmente
usando procesamiento de imágenes clásico.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional
from scipy import stats as scipy_stats
from scipy.ndimage import label as ndimage_label

warnings.filterwarnings('ignore')


# ============================================================================
# Constantes
# ============================================================================
RANDOM_STATE = 42
N_BINS_HISTOGRAM = 16
GRID_SIZE = 4  # Cuadrícula 4x4 para características por zonas
IMG_SIZE = 28


# ============================================================================
# A. CARACTERÍSTICAS DE INTENSIDAD
# ============================================================================
def extract_intensity_features(image_2d: np.ndarray) -> Dict[str, float]:
    """
    Extrae características basadas en la intensidad de píxeles.
    
    Analiza la distribución estadística de los valores de intensidad
    de la imagen para capturar información sobre brillo, contraste
    y distribución de grises.
    
    Args:
        image_2d: Imagen 2D normalizada [0, 1], shape (28, 28).
    
    Returns:
        Diccionario con las características de intensidad.
    """
    features = {}
    flat = image_2d.flatten()
    
    # Estadísticas básicas de intensidad
    features['intensity_mean'] = float(np.mean(flat))
    features['intensity_std'] = float(np.std(flat))
    features['intensity_min'] = float(np.min(flat))
    features['intensity_max'] = float(np.max(flat))
    features['intensity_median'] = float(np.median(flat))
    features['intensity_p25'] = float(np.percentile(flat, 25))
    features['intensity_p75'] = float(np.percentile(flat, 75))
    features['intensity_range'] = float(np.ptp(flat))
    
    # Brillo y contraste
    features['brightness'] = float(np.mean(image_2d))
    features['contrast'] = float(np.std(image_2d))
    
    # Entropía de Shannon
    hist, _ = np.histogram(flat, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    features['entropy'] = float(scipy_stats.entropy(hist))
    
    # Proporción de píxeles oscuros y claros
    features['dark_pixel_ratio'] = float(np.mean(flat < 0.3))
    features['bright_pixel_ratio'] = float(np.mean(flat > 0.7))
    
    # Histograma de intensidad con N_BINS_HISTOGRAM bins
    hist_values, _ = np.histogram(flat, bins=N_BINS_HISTOGRAM, range=(0, 1))
    hist_normalized = hist_values.astype(float) / hist_values.sum()
    for b in range(N_BINS_HISTOGRAM):
        features[f'histogram_bin_{b}'] = float(hist_normalized[b])
    
    # Medias por zonas (cuadrícula GRID_SIZE x GRID_SIZE)
    h, w = image_2d.shape
    zone_h = h // GRID_SIZE
    zone_w = w // GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            zone = image_2d[r*zone_h:(r+1)*zone_h, c*zone_w:(c+1)*zone_w]
            features[f'zone_mean_{r}_{c}'] = float(np.mean(zone))
    
    return features


# ============================================================================
# B. CARACTERÍSTICAS DE FORMA Y BORDES
# ============================================================================
def extract_shape_features(image_2d: np.ndarray) -> Dict[str, float]:
    """
    Extrae características de forma y bordes del objeto en la imagen.
    
    Usa umbralización de Otsu para segmentar el objeto y luego calcula
    propiedades geométricas como área, perímetro, bounding box, momentos
    de Hu y densidad de bordes Canny.
    
    Args:
        image_2d: Imagen 2D normalizada [0, 1], shape (28, 28).
    
    Returns:
        Diccionario con las características de forma y bordes.
    """
    from skimage.filters import threshold_otsu
    from skimage.feature import canny
    from skimage.measure import moments, moments_central, moments_hu
    
    features = {}
    h, w = image_2d.shape
    total_pixels = h * w
    
    # --- Segmentación con Otsu ---
    try:
        if image_2d.max() - image_2d.min() < 1e-10:
            thresh = 0.5
        else:
            thresh = threshold_otsu(image_2d)
        binary = image_2d > thresh
    except Exception:
        binary = image_2d > 0.5
    
    # Área del objeto
    object_area = int(np.sum(binary))
    features['object_area'] = float(object_area)
    features['object_area_ratio'] = float(object_area / total_pixels)
    
    # Perímetro (aproximación: píxeles de borde del objeto)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary)
    perimeter_mask = binary & ~eroded
    features['perimeter'] = float(np.sum(perimeter_mask))
    
    # Bounding box
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
    else:
        y_min, x_min = 0, 0
        bbox_h, bbox_w = h, w
    
    features['bbox_x'] = float(x_min)
    features['bbox_y'] = float(y_min)
    features['bbox_width'] = float(bbox_w)
    features['bbox_height'] = float(bbox_h)
    
    # Relación de aspecto
    if bbox_w > 0:
        features['aspect_ratio'] = float(bbox_h / bbox_w)
    else:
        features['aspect_ratio'] = 1.0
    
    # Extensión (área del objeto / área del bounding box)
    bbox_area = bbox_h * bbox_w
    if bbox_area > 0:
        features['extent'] = float(object_area / bbox_area)
    else:
        features['extent'] = 0.0
    
    # Centroide
    if object_area > 0:
        cy, cx = np.where(binary)
        features['centroid_x'] = float(np.mean(cx))
        features['centroid_y'] = float(np.mean(cy))
    else:
        features['centroid_x'] = float(w / 2)
        features['centroid_y'] = float(h / 2)
    
    # Momentos de Hu (7 momentos invariantes a transformaciones)
    try:
        m = moments(binary.astype(float))
        cr = m[0, 1] / m[0, 0] if m[0, 0] > 0 else 0
        cc = m[1, 0] / m[0, 0] if m[0, 0] > 0 else 0
        mc = moments_central(binary.astype(float), center=(cc, cr))
        hu = moments_hu(mc)
        for i in range(7):
            # Aplicar log transform para estabilizar valores
            val = hu[i]
            if val != 0:
                features[f'hu_moment_{i}'] = float(-np.sign(val) * np.log10(abs(val) + 1e-20))
            else:
                features[f'hu_moment_{i}'] = 0.0
    except Exception:
        for i in range(7):
            features[f'hu_moment_{i}'] = 0.0
    
    # --- Bordes Canny ---
    try:
        edges = canny(image_2d, sigma=1.0)
        n_edge_pixels = int(np.sum(edges))
        features['canny_edge_density'] = float(n_edge_pixels / total_pixels)
        features['n_edge_pixels'] = float(n_edge_pixels)
    except Exception:
        features['canny_edge_density'] = 0.0
        features['n_edge_pixels'] = 0.0
    
    # Densidad de bordes por zonas (cuadrícula GRID_SIZE x GRID_SIZE)
    try:
        zone_h = h // GRID_SIZE
        zone_w = w // GRID_SIZE
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                zone = edges[r*zone_h:(r+1)*zone_h, c*zone_w:(c+1)*zone_w]
                features[f'edge_zone_{r}_{c}'] = float(np.mean(zone))
    except Exception:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                features[f'edge_zone_{r}_{c}'] = 0.0
    
    return features


# ============================================================================
# C. CARACTERÍSTICAS DE TEXTURA
# ============================================================================
def extract_texture_features(image_2d: np.ndarray) -> Dict[str, float]:
    """
    Extrae características de textura de la imagen.
    
    Usa LBP (Local Binary Pattern), GLCM (Gray-Level Co-occurrence Matrix)
    y HOG (Histogram of Oriented Gradients) para capturar patrones
    texturales que diferencian las prendas.
    
    Args:
        image_2d: Imagen 2D normalizada [0, 1], shape (28, 28).
    
    Returns:
        Diccionario con las características de textura.
    """
    from skimage.feature import local_binary_pattern, hog
    from skimage.feature import graycomatrix, graycoprops
    
    features = {}
    
    # --- LBP (Local Binary Pattern) ---
    try:
        P, R = 8, 1  # 8 vecinos, radio 1
        lbp = local_binary_pattern(image_2d, P, R, method='uniform')
        # Histograma normalizado de LBP (P+2 bins para método 'uniform')
        n_bins_lbp = P + 2  # 10 bins
        lbp_hist, _ = np.histogram(lbp, bins=n_bins_lbp, range=(0, n_bins_lbp), density=True)
        for b in range(min(n_bins_lbp, 10)):
            features[f'lbp_hist_{b}'] = float(lbp_hist[b])
    except Exception:
        for b in range(10):
            features[f'lbp_hist_{b}'] = 0.0
    
    # --- GLCM (Gray-Level Co-occurrence Matrix) ---
    try:
        # Cuantizar imagen a 8 niveles
        img_quantized = (image_2d * 7).astype(np.uint8)
        
        glcm = graycomatrix(img_quantized, distances=[1], angles=[0],
                            levels=8, symmetric=True, normed=True)
        
        features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
        features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
        features['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        features['glcm_correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
        features['glcm_dissimilarity'] = float(graycoprops(glcm, 'dissimilarity')[0, 0])
        features['glcm_ASM'] = float(graycoprops(glcm, 'ASM')[0, 0])
    except Exception:
        for prop in ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']:
            features[f'glcm_{prop}'] = 0.0
    
    # --- HOG (Histogram of Oriented Gradients) ---
    try:
        hog_features = hog(image_2d, orientations=9, pixels_per_cell=(7, 7),
                           cells_per_block=(2, 2), feature_vector=True)
        
        # Resumen estadístico del descriptor HOG
        features['hog_mean'] = float(np.mean(hog_features))
        features['hog_std'] = float(np.std(hog_features))
        features['hog_max'] = float(np.max(hog_features))
        features['hog_min'] = float(np.min(hog_features))
        features['hog_median'] = float(np.median(hog_features))
    except Exception:
        for stat in ['mean', 'std', 'max', 'min', 'median']:
            features[f'hog_{stat}'] = 0.0
    
    # --- Entropía global ---
    try:
        from skimage.measure import shannon_entropy
        features['texture_entropy'] = float(shannon_entropy(image_2d))
    except Exception:
        features['texture_entropy'] = 0.0
    
    return features


# ============================================================================
# D. EXTRACCIÓN COMBINADA
# ============================================================================
def extract_all_features(image_2d: np.ndarray) -> Dict[str, float]:
    """
    Combina todas las familias de características para una imagen.
    
    Extrae características de intensidad, forma/bordes y textura
    y las combina en un solo diccionario.
    
    Args:
        image_2d: Imagen 2D normalizada [0, 1], shape (28, 28).
    
    Returns:
        Diccionario con todas las características combinadas.
    """
    features = {}
    features.update(extract_intensity_features(image_2d))
    features.update(extract_shape_features(image_2d))
    features.update(extract_texture_features(image_2d))
    return features


def extract_features_batch(X: np.ndarray, n_jobs: int = 1) -> pd.DataFrame:
    """
    Extrae características para un lote completo de imágenes.
    
    Procesa todas las imágenes del array, normalizándolas y extrayendo
    el vector completo de características para cada una.
    
    Args:
        X: Array de imágenes, shape (n, 784), dtype uint8.
        n_jobs: Número de trabajos paralelos (1 = secuencial).
    
    Returns:
        DataFrame donde cada fila es una imagen y cada columna una característica.
    """
    n_images = len(X)
    print(f"\n  Extrayendo características de {n_images:,} imágenes...")
    
    # Normalizar y remodelar
    X_norm = X.astype(np.float64) / 255.0
    X_2d = X_norm.reshape(-1, 28, 28)
    
    if n_jobs > 1:
        # Procesamiento paralelo con joblib
        try:
            from joblib import Parallel, delayed
            
            def _extract_single(i):
                return extract_all_features(X_2d[i])
            
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_extract_single)(i) for i in range(n_images)
            )
            print(f"  ✓ Extracción completada ({n_images:,} imágenes)")
            return pd.DataFrame(results)
        except ImportError:
            print("  ⚠ joblib no disponible, usando extracción secuencial.")
    
    # Procesamiento secuencial
    all_features = []
    for i in range(n_images):
        features = extract_all_features(X_2d[i])
        all_features.append(features)
        
        if (i + 1) % 500 == 0 or i == n_images - 1:
            pct = 100 * (i + 1) / n_images
            print(f"    Progreso: {i+1:,}/{n_images:,} ({pct:.1f}%)")
    
    df = pd.DataFrame(all_features)
    print(f"  ✓ Extracción completada: {df.shape[1]} características x {df.shape[0]:,} imágenes")
    
    return df


def get_feature_documentation() -> pd.DataFrame:
    """
    Genera documentación de todas las características extraídas.
    
    Returns:
        DataFrame con columnas: nombre, familia, descripción.
    """
    docs = []
    
    # Intensidad
    intensity_features = [
        ('intensity_mean', 'Intensidad', 'Media de intensidad de todos los píxeles'),
        ('intensity_std', 'Intensidad', 'Desviación estándar de intensidad'),
        ('intensity_min', 'Intensidad', 'Valor mínimo de intensidad'),
        ('intensity_max', 'Intensidad', 'Valor máximo de intensidad'),
        ('intensity_median', 'Intensidad', 'Mediana de intensidad'),
        ('intensity_p25', 'Intensidad', 'Percentil 25 de intensidad'),
        ('intensity_p75', 'Intensidad', 'Percentil 75 de intensidad'),
        ('intensity_range', 'Intensidad', 'Rango de intensidad (max - min)'),
        ('brightness', 'Intensidad', 'Brillo promedio de la imagen'),
        ('contrast', 'Intensidad', 'Contraste (desviación estándar de intensidad)'),
        ('entropy', 'Intensidad', 'Entropía de Shannon de la distribución de intensidad'),
        ('dark_pixel_ratio', 'Intensidad', 'Proporción de píxeles oscuros (< 0.3)'),
        ('bright_pixel_ratio', 'Intensidad', 'Proporción de píxeles claros (> 0.7)'),
    ]
    
    for b in range(N_BINS_HISTOGRAM):
        intensity_features.append(
            (f'histogram_bin_{b}', 'Intensidad', f'Bin {b} del histograma de intensidad ({N_BINS_HISTOGRAM} bins)')
        )
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            intensity_features.append(
                (f'zone_mean_{r}_{c}', 'Intensidad', f'Media de intensidad en zona ({r},{c}) de cuadrícula {GRID_SIZE}x{GRID_SIZE}')
            )
    
    # Forma y bordes
    shape_features = [
        ('object_area', 'Forma', 'Área del objeto segmentado (píxeles de primer plano)'),
        ('object_area_ratio', 'Forma', 'Proporción del área del objeto respecto al total'),
        ('perimeter', 'Forma', 'Perímetro aproximado del objeto (píxeles de borde)'),
        ('bbox_x', 'Forma', 'Coordenada X del bounding box'),
        ('bbox_y', 'Forma', 'Coordenada Y del bounding box'),
        ('bbox_width', 'Forma', 'Ancho del bounding box'),
        ('bbox_height', 'Forma', 'Alto del bounding box'),
        ('aspect_ratio', 'Forma', 'Relación alto/ancho del bounding box'),
        ('extent', 'Forma', 'Extensión: área del objeto / área del bounding box'),
        ('centroid_x', 'Forma', 'Coordenada X del centroide del objeto'),
        ('centroid_y', 'Forma', 'Coordenada Y del centroide del objeto'),
    ]
    
    for i in range(7):
        shape_features.append(
            (f'hu_moment_{i}', 'Forma', f'Momento de Hu #{i+1} (invariante a transformaciones)')
        )
    
    shape_features.extend([
        ('canny_edge_density', 'Bordes', 'Densidad de bordes Canny (proporción de píxeles de borde)'),
        ('n_edge_pixels', 'Bordes', 'Número total de píxeles de borde Canny'),
    ])
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            shape_features.append(
                (f'edge_zone_{r}_{c}', 'Bordes', f'Densidad de bordes en zona ({r},{c}) de cuadrícula {GRID_SIZE}x{GRID_SIZE}')
            )
    
    # Textura
    texture_features = []
    for b in range(10):
        texture_features.append(
            (f'lbp_hist_{b}', 'Textura', f'Bin {b} del histograma LBP (Local Binary Pattern)')
        )
    
    texture_features.extend([
        ('glcm_contrast', 'Textura', 'Contraste GLCM (diferencia de niveles de gris entre vecinos)'),
        ('glcm_homogeneity', 'Textura', 'Homogeneidad GLCM (cercanía a la diagonal)'),
        ('glcm_energy', 'Textura', 'Energía GLCM (uniformidad de la textura)'),
        ('glcm_correlation', 'Textura', 'Correlación GLCM (dependencia lineal entre niveles de gris)'),
        ('glcm_dissimilarity', 'Textura', 'Disimilitud GLCM (diferencia promedio entre píxeles vecinos)'),
        ('glcm_ASM', 'Textura', 'ASM GLCM (Angular Second Moment - uniformidad)'),
        ('hog_mean', 'Textura', 'Media del descriptor HOG'),
        ('hog_std', 'Textura', 'Desviación estándar del descriptor HOG'),
        ('hog_max', 'Textura', 'Máximo del descriptor HOG'),
        ('hog_min', 'Textura', 'Mínimo del descriptor HOG'),
        ('hog_median', 'Textura', 'Mediana del descriptor HOG'),
        ('texture_entropy', 'Textura', 'Entropía de Shannon de la imagen (complejidad textural)'),
    ])
    
    all_docs = intensity_features + shape_features + texture_features
    
    df = pd.DataFrame(all_docs, columns=['Nombre', 'Familia', 'Descripción'])
    return df
