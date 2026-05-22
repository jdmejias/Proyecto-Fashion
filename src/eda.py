# -*- coding: utf-8 -*-
"""
Módulo de Análisis Exploratorio de Datos (EDA) - Proyecto Fashion-MNIST.

Funciones para analizar y visualizar el dataset Fashion-MNIST:
distribuciones, muestras, histogramas, brillo, contraste, texturas,
bordes, detección de clases similares y outliers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from typing import Dict, Optional
from pathlib import Path

from .utils import get_project_root


# ============================================================================
# Estilo global de gráficas
# ============================================================================
def _set_style():
    """Establece un estilo visual consistente para las gráficas."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            plt.style.use('ggplot')


# Paleta de colores para las clases
CLASS_COLORS = {
    0: '#FF6B6B',   # T-shirt/top - Rojo coral
    1: '#4ECDC4',   # Trouser - Turquesa
    3: '#45B7D1',   # Dress - Azul claro
    6: '#F7DC6F',   # Shirt - Amarillo
    8: '#BB8FCE',   # Bag - Púrpura
}


def plot_class_distribution(y: np.ndarray, class_names: dict, 
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Gráfica de barras con la distribución de clases (conteo y porcentaje).
    
    Args:
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar. Si None, se muestra.
    
    Returns:
        Figura de matplotlib.
    """
    _set_style()
    classes = sorted(class_names.keys())
    counts = [np.sum(y == c) for c in classes]
    total = len(y)
    percentages = [100 * c / total for c in counts]
    labels = [f"{class_names[c]}\n(Clase {c})" for c in classes]
    colors = [CLASS_COLORS.get(c, '#888888') for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Añadir etiquetas con conteo y porcentaje
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + total*0.005,
                f'{count:,}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Distribución de Clases en el Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Número de Imágenes', fontsize=12)
    ax.set_xlabel('Clase', fontsize=12)
    ax.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_sample_images(X: np.ndarray, y: np.ndarray, class_names: dict,
                       n_per_class: int = 5, 
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Muestra una cuadrícula de imágenes de ejemplo por cada clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        n_per_class: Número de muestras por clase.
        save_path: Ruta donde guardar.
    
    Returns:
        Figura de matplotlib.
    """
    _set_style()
    classes = sorted(class_names.keys())
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_classes, n_per_class, 
                              figsize=(2 * n_per_class, 2.5 * n_classes))
    fig.suptitle('Muestras Representativas por Clase', fontsize=14, fontweight='bold')
    
    for i, cls_id in enumerate(classes):
        indices = np.where(y == cls_id)[0]
        sample_idx = np.random.choice(indices, size=min(n_per_class, len(indices)), replace=False)
        
        for j in range(n_per_class):
            ax = axes[i, j] if n_classes > 1 else axes[j]
            if j < len(sample_idx):
                img = X[sample_idx[j]].reshape(28, 28)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'{class_names[cls_id]}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_intensity_histograms(X: np.ndarray, y: np.ndarray, class_names: dict,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Histogramas de distribución de intensidad de píxeles por clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    
    Returns:
        Figura de matplotlib.
    """
    _set_style()
    classes = sorted(class_names.keys())
    
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    fig.suptitle('Distribución de Intensidad de Píxeles por Clase', 
                 fontsize=14, fontweight='bold')
    
    for i, cls_id in enumerate(classes):
        ax = axes[i] if len(classes) > 1 else axes
        mask = y == cls_id
        # Tomar una muestra para no sobrecargar
        sample = X[mask][:500].flatten()
        
        color = CLASS_COLORS.get(cls_id, '#888888')
        ax.hist(sample, bins=50, color=color, alpha=0.7, density=True, edgecolor='white')
        ax.set_title(f'{class_names[cls_id]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Intensidad (0-255)', fontsize=9)
        ax.set_ylabel('Densidad', fontsize=9)
        ax.set_xlim(0, 255)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_brightness_analysis(X: np.ndarray, y: np.ndarray, class_names: dict,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Boxplot de brillo promedio (media de intensidad) por clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    classes = sorted(class_names.keys())
    
    brightness_data = []
    labels = []
    colors = []
    
    for cls_id in classes:
        mask = y == cls_id
        brightness = X[mask].mean(axis=1) / 255.0
        brightness_data.append(brightness)
        labels.append(f'{class_names[cls_id]}\n(Clase {cls_id})')
        colors.append(CLASS_COLORS.get(cls_id, '#888888'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(brightness_data, labels=labels, patch_artist=True,
                     widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='gold'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Análisis de Brillo Promedio por Clase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Brillo Promedio (0-1)', fontsize=12)
    ax.set_xlabel('Clase', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_contrast_analysis(X: np.ndarray, y: np.ndarray, class_names: dict,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Boxplot de contraste (desviación estándar de intensidad) por clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    classes = sorted(class_names.keys())
    
    contrast_data = []
    labels = []
    colors = []
    
    for cls_id in classes:
        mask = y == cls_id
        contrast = X[mask].astype(float).std(axis=1) / 255.0
        contrast_data.append(contrast)
        labels.append(f'{class_names[cls_id]}\n(Clase {cls_id})')
        colors.append(CLASS_COLORS.get(cls_id, '#888888'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(contrast_data, labels=labels, patch_artist=True,
                     widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='gold'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Análisis de Contraste por Clase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Contraste (Desviación Estándar, 0-1)', fontsize=12)
    ax.set_xlabel('Clase', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_texture_analysis(X: np.ndarray, y: np.ndarray, class_names: dict,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Análisis de textura: muestra la varianza local promedio por clase.
    
    Computa la varianza local con una ventana deslizante como proxy
    de complejidad textural de cada clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    from scipy.ndimage import uniform_filter
    
    classes = sorted(class_names.keys())
    
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    fig.suptitle('Análisis de Textura: Varianza Local Promedio por Clase', 
                 fontsize=14, fontweight='bold')
    
    for i, cls_id in enumerate(classes):
        ax = axes[i] if len(classes) > 1 else axes
        mask = y == cls_id
        # Promediar imágenes de la clase
        sample = X[mask][:200].reshape(-1, 28, 28).astype(float) / 255.0
        
        # Varianza local promedio
        variances = []
        for img in sample:
            local_mean = uniform_filter(img, size=3)
            local_sqr_mean = uniform_filter(img**2, size=3)
            local_var = local_sqr_mean - local_mean**2
            variances.append(local_var.mean())
        
        color = CLASS_COLORS.get(cls_id, '#888888')
        ax.hist(variances, bins=30, color=color, alpha=0.7, edgecolor='white')
        ax.set_title(f'{class_names[cls_id]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Varianza Local Promedio', fontsize=9)
        ax.set_ylabel('Frecuencia', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_edge_analysis(X: np.ndarray, y: np.ndarray, class_names: dict,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Análisis de bordes: densidad de bordes Canny por clase.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    from skimage.feature import canny
    
    classes = sorted(class_names.keys())
    
    edge_data = []
    labels = []
    colors_list = []
    
    for cls_id in classes:
        mask = y == cls_id
        sample = X[mask][:300].reshape(-1, 28, 28).astype(float) / 255.0
        
        edge_densities = []
        for img in sample:
            edges = canny(img, sigma=1.0)
            edge_densities.append(edges.mean())
        
        edge_data.append(edge_densities)
        labels.append(f'{class_names[cls_id]}\n(Clase {cls_id})')
        colors_list.append(CLASS_COLORS.get(cls_id, '#888888'))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(edge_data, labels=labels, patch_artist=True,
                     widths=0.6, showmeans=True,
                     meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='gold'))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Densidad de Bordes (Canny) por Clase', fontsize=14, fontweight='bold')
    ax.set_ylabel('Densidad de Bordes', fontsize=12)
    ax.set_xlabel('Clase', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    return fig


def detect_similar_classes(X: np.ndarray, y: np.ndarray, class_names: dict,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Detecta clases visualmente similares usando imágenes promedio y correlación.
    
    Calcula la imagen promedio de cada clase y computa una matriz de
    similitud basada en correlación de Pearson entre los promedios.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    classes = sorted(class_names.keys())
    n_classes = len(classes)
    
    # Calcular imágenes promedio
    avg_images = {}
    for cls_id in classes:
        mask = y == cls_id
        avg_images[cls_id] = X[mask].astype(float).mean(axis=0)
    
    # Matriz de similitud (correlación)
    sim_matrix = np.zeros((n_classes, n_classes))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            corr = np.corrcoef(avg_images[c1], avg_images[c2])[0, 1]
            sim_matrix[i, j] = corr
    
    # Figura con dos paneles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Imágenes promedio
    for i, cls_id in enumerate(classes):
        ax_inset = fig.add_axes([0.05 + i * 0.08, 0.55, 0.07, 0.35])
        ax_inset.imshow(avg_images[cls_id].reshape(28, 28), cmap='gray')
        ax_inset.set_title(f'{class_names[cls_id]}', fontsize=7, fontweight='bold')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
    
    ax1.set_visible(False)
    
    # Panel 2: Matriz de similitud
    labels = [class_names[c] for c in classes]
    im = ax2.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title('Matriz de Similitud entre Clases\n(Correlación de Pearson)', 
                   fontsize=13, fontweight='bold')
    
    # Añadir valores
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if sim_matrix[i, j] < 0.5 else 'black'
            ax2.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax2, shrink=0.8, label='Correlación')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    # Reportar pares más similares
    print("\n  Pares de clases más similares (excluyendo diagonal):")
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            if sim_matrix[i, j] > 0.7:
                print(f"    ⚠ {class_names[classes[i]]} ↔ {class_names[classes[j]]}: "
                      f"correlación = {sim_matrix[i, j]:.3f}")
    
    return fig


def detect_outliers(X: np.ndarray, y: np.ndarray, class_names: dict,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Detecta posibles outliers visuales basándose en brillo extremo
    y pocos píxeles activos.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    _set_style()
    classes = sorted(class_names.keys())
    
    # Calcular estadísticas para detectar outliers
    brightness = X.astype(float).mean(axis=1) / 255.0
    active_pixels = (X > 20).sum(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Detección de Posibles Outliers Visuales', fontsize=14, fontweight='bold')
    
    # Panel 1: Brillo vs píxeles activos
    for cls_id in classes:
        mask = y == cls_id
        color = CLASS_COLORS.get(cls_id, '#888888')
        axes[0].scatter(brightness[mask][:500], active_pixels[mask][:500], 
                        alpha=0.3, s=10, color=color, label=class_names[cls_id])
    
    axes[0].set_xlabel('Brillo Promedio', fontsize=11)
    axes[0].set_ylabel('Píxeles Activos (>20)', fontsize=11)
    axes[0].set_title('Brillo vs Píxeles Activos', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, markerscale=3)
    
    # Panel 2: Imágenes con brillo extremo
    # Encontrar las 5 imágenes más oscuras y las 5 más brillantes
    darkest_idx = np.argsort(brightness)[:5]
    brightest_idx = np.argsort(brightness)[-5:]
    
    for i, idx in enumerate(np.concatenate([darkest_idx, brightest_idx])):
        row = 0 if i < 5 else 1
        col = i if i < 5 else i - 5
        ax_inset = fig.add_axes([0.55 + col * 0.08, 0.55 - row * 0.4, 0.07, 0.3])
        ax_inset.imshow(X[idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        label = f"B={brightness[idx]:.2f}"
        ax_inset.set_title(label, fontsize=6)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
    
    axes[1].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Figura guardada: {save_path}")
        plt.close(fig)
    
    # Reportar estadísticas de outliers
    q1 = np.percentile(brightness, 5)
    q3 = np.percentile(brightness, 95)
    n_outliers_dark = np.sum(brightness < q1 * 0.5)
    n_outliers_bright = np.sum(brightness > q3 * 1.5)
    print(f"\n  Outliers detectados:")
    print(f"    → Muy oscuros (brillo < {q1*0.5:.3f}): {n_outliers_dark}")
    print(f"    → Muy brillantes (brillo > {q3*1.5:.3f}): {n_outliers_bright}")
    
    return fig


def dataset_summary(X: np.ndarray, y: np.ndarray, 
                    class_names: dict) -> pd.DataFrame:
    """
    Genera un resumen estadístico del dataset.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas.
        class_names: Diccionario {id: nombre}.
    
    Returns:
        DataFrame con estadísticas por clase.
    """
    classes = sorted(class_names.keys())
    
    summary_data = []
    for cls_id in classes:
        mask = y == cls_id
        X_cls = X[mask].astype(float)
        
        summary_data.append({
            'Clase': cls_id,
            'Nombre': class_names[cls_id],
            'N_imágenes': mask.sum(),
            'Porcentaje': f"{100 * mask.sum() / len(y):.1f}%",
            'Brillo_medio': f"{X_cls.mean() / 255:.3f}",
            'Brillo_std': f"{X_cls.mean(axis=1).std() / 255:.3f}",
            'Contraste_medio': f"{X_cls.std(axis=1).mean() / 255:.3f}",
            'Píxeles_activos_medio': f"{(X_cls > 20).sum(axis=1).mean():.0f}",
        })
    
    # Totales
    summary_data.append({
        'Clase': '-',
        'Nombre': 'TOTAL',
        'N_imágenes': len(y),
        'Porcentaje': '100.0%',
        'Brillo_medio': f"{X.astype(float).mean() / 255:.3f}",
        'Brillo_std': f"{X.astype(float).mean(axis=1).std() / 255:.3f}",
        'Contraste_medio': f"{X.astype(float).std(axis=1).mean() / 255:.3f}",
        'Píxeles_activos_medio': f"{(X > 20).sum(axis=1).mean():.0f}",
    })
    
    df = pd.DataFrame(summary_data)
    return df
