# -*- coding: utf-8 -*-
"""
Módulo de preprocesamiento de imágenes - Proyecto Fashion-MNIST.

Funciones para normalizar, segmentar y preprocesar las imágenes 28x28
del dataset Fashion-MNIST antes de la extracción de características.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pathlib import Path

from .utils import get_project_root, save_figure


def normalize_images(X: np.ndarray) -> np.ndarray:
    """
    Normaliza los valores de píxeles al rango [0, 1].
    
    Args:
        X: Array de imágenes con valores uint8 [0, 255].
           Shape puede ser (n, 784) o (n, 28, 28).
    
    Returns:
        np.ndarray con valores float64 en [0, 1].
    """
    return X.astype(np.float64) / 255.0


def reshape_images(X: np.ndarray) -> np.ndarray:
    """
    Convierte imágenes de vectores 784 a matrices 28x28.
    
    Args:
        X: Array de imágenes, shape (n, 784).
    
    Returns:
        np.ndarray con shape (n, 28, 28).
    """
    if X.ndim == 2 and X.shape[1] == 784:
        return X.reshape(-1, 28, 28)
    elif X.ndim == 3 and X.shape[1:] == (28, 28):
        return X  # Ya está en formato 2D
    else:
        raise ValueError(f"Shape no esperado: {X.shape}. Se esperaba (n, 784) o (n, 28, 28).")


def apply_gaussian_smoothing(images: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica suavizado gaussiano a las imágenes.
    
    El suavizado gaussiano reduce el ruido manteniendo las estructuras
    principales de la imagen. Es útil antes de la segmentación.
    
    Args:
        images: Array de imágenes, shape (n, 28, 28), valores en [0, 1].
        sigma: Desviación estándar del kernel gaussiano.
    
    Returns:
        np.ndarray con imágenes suavizadas.
    """
    from scipy.ndimage import gaussian_filter
    
    smoothed = np.zeros_like(images)
    for i in range(len(images)):
        smoothed[i] = gaussian_filter(images[i], sigma=sigma)
    
    return smoothed


def apply_otsu_threshold(images: np.ndarray) -> np.ndarray:
    """
    Aplica umbralización de Otsu para segmentar el objeto del fondo.
    
    El método de Otsu encuentra automáticamente el umbral óptimo que
    minimiza la varianza intra-clase, separando el objeto (prenda)
    del fondo oscuro.
    
    Args:
        images: Array de imágenes, shape (n, 28, 28), valores en [0, 1].
    
    Returns:
        np.ndarray con máscaras binarias, shape (n, 28, 28).
    """
    from skimage.filters import threshold_otsu
    
    masks = np.zeros_like(images, dtype=bool)
    
    for i in range(len(images)):
        img = images[i]
        if img.max() - img.min() < 1e-10:
            # Imagen completamente uniforme
            masks[i] = img > 0.5
        else:
            try:
                thresh = threshold_otsu(img)
                masks[i] = img > thresh
            except Exception:
                masks[i] = img > 0.5
    
    return masks


def get_bounding_box(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Obtiene el bounding box del objeto en una máscara binaria.
    
    Args:
        binary_mask: Máscara binaria 2D (28x28).
    
    Returns:
        Tuple (y_min, x_min, height, width) del bounding box.
        Si no hay píxeles activos, retorna (0, 0, 28, 28).
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not rows.any():
        return (0, 0, binary_mask.shape[0], binary_mask.shape[1])
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    
    return (int(y_min), int(x_min), int(height), int(width))


def segment_images(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completo de segmentación: normalizar, remodelar, umbralizar.
    
    Args:
        X: Array de imágenes, shape (n, 784), valores uint8.
    
    Returns:
        Tuple con:
            - images_norm: Imágenes normalizadas [0,1], shape (n, 28, 28).
            - masks: Máscaras binarias de Otsu, shape (n, 28, 28).
            - images_segmented: Imágenes segmentadas (imagen * máscara).
    """
    # Normalizar
    X_norm = normalize_images(X)
    
    # Remodelar a 28x28
    images = reshape_images(X_norm)
    
    # Aplicar suavizado ligero antes de segmentar
    images_smooth = apply_gaussian_smoothing(images, sigma=0.5)
    
    # Obtener máscaras binarias con Otsu
    masks = apply_otsu_threshold(images_smooth)
    
    # Aplicar máscara sobre la imagen original normalizada
    images_segmented = images * masks
    
    return images, masks, images_segmented


def compare_original_vs_segmented(X: np.ndarray, y: np.ndarray, 
                                   class_names: dict,
                                   save_path: Optional[str] = None) -> None:
    """
    Genera una comparación visual entre imágenes originales y segmentadas.
    
    Muestra una cuadrícula con la imagen original, la máscara de Otsu
    y la imagen segmentada para una muestra de cada clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar la figura. Si None, se muestra.
    """
    images, masks, segmented = segment_images(X)
    
    classes = sorted(class_names.keys())
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_classes, 3, figsize=(9, 3 * n_classes))
    fig.suptitle('Comparación: Original vs Segmentada (Otsu)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    col_titles = ['Original', 'Máscara Otsu', 'Segmentada']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight='bold')
    
    for i, cls_id in enumerate(classes):
        idx = np.where(y == cls_id)[0][0]
        
        axes[i, 0].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f'{class_names[cls_id]}', fontsize=10, fontweight='bold')
        
        axes[i, 1].imshow(masks[idx], cmap='gray')
        
        axes[i, 2].imshow(segmented[idx], cmap='gray', vmin=0, vmax=1)
        
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    else:
        plt.show()
