# -*- coding: utf-8 -*-
"""
Módulo de evaluación de modelos - Proyecto Fashion-MNIST.

Funciones para generar matrices de confusión, métricas detalladas,
tablas comparativas y visualizaciones de predicciones correctas
e incorrectas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score)

from .utils import get_project_root


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: dict, model_name: str = '',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Genera una matriz de confusión visual con mapa de calor.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Diccionario {id: nombre}.
        model_name: Nombre del modelo para el título.
        save_path: Ruta donde guardar la figura.
    
    Returns:
        Figura de matplotlib.
    """
    labels = sorted(class_names.keys())
    label_names = [class_names[l] for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor='gray')
    
    # Añadir porcentajes como texto secundario
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.75, f'({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=7, color='gray')
    
    title = 'Matriz de Confusión'
    if model_name:
        title += f' - {model_name}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Etiqueta Real', fontsize=11)
    ax.set_xlabel('Etiqueta Predicha', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Matriz de confusión guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_confusion_matrices_grid(results_list: List[Dict], y_test: np.ndarray,
                                  class_names: dict,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Genera una cuadrícula 3x3 con matrices de confusión de todos los modelos.
    
    Args:
        results_list: Lista de diccionarios de resultados (de train_and_evaluate).
        y_test: Etiquetas verdaderas de prueba.
        class_names: Diccionario {id: nombre}.
        save_path: Ruta donde guardar.
    """
    labels = sorted(class_names.keys())
    label_names = [class_names[l] for l in labels]
    
    n_models = len([r for r in results_list if r.get('success', False)])
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle('Matrices de Confusión - Todos los Modelos', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes_flat = axes.flatten() if n_rows > 1 else (axes if n_cols > 1 else [axes])
    
    idx = 0
    for r in results_list:
        if not r.get('success', False):
            continue
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        cm = confusion_matrix(y_test, r['y_pred'], labels=labels)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=label_names, yticklabels=label_names,
                    linewidths=0.3, cbar=False)
        ax.set_title(r['model_name'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Real', fontsize=9)
        ax.set_xlabel('Predicho', fontsize=9)
        ax.tick_params(labelsize=7)
        idx += 1
    
    # Ocultar ejes sobrantes
    for i in range(idx, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Grid de matrices guardado: {save_path}")
        plt.close(fig)
    
    return fig


def get_full_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: dict) -> Dict[str, Any]:
    """
    Calcula métricas completas de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Diccionario {id: nombre}.
    
    Returns:
        Diccionario con todas las métricas.
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'classification_report': classification_report(y_true, y_pred, 
                                                        target_names=[class_names[c] for c in sorted(class_names.keys())],
                                                        zero_division=0),
    }
    
    # Métricas por clase
    labels = sorted(class_names.keys())
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, cls_id in enumerate(labels):
        metrics['per_class'][class_names[cls_id]] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
        }
    
    return metrics


def create_comparison_table(results_list: List[Dict]) -> pd.DataFrame:
    """
    Crea una tabla comparativa de todos los modelos evaluados.
    
    Args:
        results_list: Lista de diccionarios de resultados.
    
    Returns:
        DataFrame con comparación ordenada por F1-macro.
    """
    data = []
    for r in results_list:
        if r.get('success', False):
            data.append({
                'Modelo': r['model_name'],
                'Accuracy': round(r['accuracy'], 4),
                'Precision_macro': round(r.get('precision_macro', 0), 4),
                'Recall_macro': round(r.get('recall_macro', 0), 4),
                'F1_macro': round(r['f1_macro'], 4),
                'F1_weighted': round(r.get('f1_weighted', 0), 4),
                'Tiempo_entrenamiento (s)': round(r['train_time'], 3),
                'Tiempo_prediccion (s)': round(r['predict_time'], 3),
            })
    
    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.sort_values('F1_macro', ascending=False).reset_index(drop=True)
    
    return df


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = 'F1_macro',
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Gráfica de barras comparando modelos en una métrica específica.
    
    Args:
        comparison_df: DataFrame de comparación (de create_comparison_table).
        metric: Columna de la métrica a comparar.
        save_path: Ruta donde guardar.
    """
    if metric not in comparison_df.columns:
        print(f"  ⚠ Métrica '{metric}' no encontrada en la tabla.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_sorted = comparison_df.sort_values(metric, ascending=True)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_sorted)))
    
    bars = ax.barh(df_sorted['Modelo'], df_sorted[metric], color=colors, 
                    edgecolor='white', linewidth=1)
    
    # Añadir valores
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    metric_name = metric.replace('_', ' ').title()
    ax.set_title(f'Comparación de Modelos: {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_xlim(0, min(1.0, df_sorted[metric].max() * 1.15))
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Comparación guardada: {save_path}")
        plt.close(fig)
    
    return fig


def plot_correct_predictions(X_test_images: np.ndarray, y_true: np.ndarray,
                              y_pred: np.ndarray, class_names: dict,
                              n: int = 10,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Muestra imágenes correctamente clasificadas.
    
    Args:
        X_test_images: Imágenes originales, shape (n, 784).
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Diccionario {id: nombre}.
        n: Número de imágenes a mostrar.
        save_path: Ruta donde guardar.
    """
    correct_mask = y_true == y_pred
    correct_idx = np.where(correct_mask)[0]
    
    sample_size = min(n, len(correct_idx))
    sample_idx = np.random.choice(correct_idx, size=sample_size, replace=False)
    
    n_cols = 5
    n_rows = (sample_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    fig.suptitle('Predicciones Correctas', fontsize=14, fontweight='bold')
    
    axes_flat = axes.flatten() if n_rows > 1 else (axes if isinstance(axes, np.ndarray) else [axes])
    
    for i, idx in enumerate(sample_idx):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        ax.imshow(X_test_images[idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'✓ {class_names.get(y_true[idx], y_true[idx])}', 
                      fontsize=9, color='green', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    for i in range(sample_size, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Predicciones correctas guardadas: {save_path}")
        plt.close(fig)
    
    return fig


def plot_incorrect_predictions(X_test_images: np.ndarray, y_true: np.ndarray,
                                y_pred: np.ndarray, class_names: dict,
                                probas: Optional[np.ndarray] = None,
                                n: int = 10,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Muestra imágenes incorrectamente clasificadas.
    
    Args:
        X_test_images: Imágenes originales, shape (n, 784).
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Diccionario {id: nombre}.
        probas: Probabilidades de predicción (opcional).
        n: Número de imágenes a mostrar.
        save_path: Ruta donde guardar.
    """
    incorrect_mask = y_true != y_pred
    incorrect_idx = np.where(incorrect_mask)[0]
    
    if len(incorrect_idx) == 0:
        print("  ✓ No hay predicciones incorrectas.")
        return None
    
    sample_size = min(n, len(incorrect_idx))
    sample_idx = np.random.choice(incorrect_idx, size=sample_size, replace=False)
    
    n_cols = 5
    n_rows = (sample_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
    fig.suptitle('Predicciones Incorrectas', fontsize=14, fontweight='bold', color='red')
    
    axes_flat = axes.flatten() if n_rows > 1 else (axes if isinstance(axes, np.ndarray) else [axes])
    
    for i, idx in enumerate(sample_idx):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        ax.imshow(X_test_images[idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        
        true_name = class_names.get(y_true[idx], str(y_true[idx]))
        pred_name = class_names.get(y_pred[idx], str(y_pred[idx]))
        
        title = f'Real: {true_name}\nPred: {pred_name}'
        if probas is not None:
            conf = probas[idx].max()
            title += f'\nConf: {conf:.2f}'
        
        ax.set_title(title, fontsize=8, color='red', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    for i in range(sample_size, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Predicciones incorrectas guardadas: {save_path}")
        plt.close(fig)
    
    return fig


def identify_confused_classes(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: dict) -> pd.DataFrame:
    """
    Identifica los pares de clases más confundidos.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        class_names: Diccionario {id: nombre}.
    
    Returns:
        DataFrame con pares de clases confundidos, ordenado por frecuencia.
    """
    labels = sorted(class_names.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'Clase_real': class_names[labels[i]],
                    'Clase_predicha': class_names[labels[j]],
                    'N_confusiones': cm[i, j],
                    'Porcentaje': f"{100 * cm[i, j] / cm[i].sum():.1f}%",
                })
    
    df = pd.DataFrame(confusions)
    if len(df) > 0:
        df = df.sort_values('N_confusiones', ascending=False).reset_index(drop=True)
    
    return df


def save_comparison_csv(comparison_df: pd.DataFrame, filename: str,
                         folder: str = 'outputs/tables') -> Path:
    """
    Guarda una tabla comparativa como CSV.
    
    Args:
        comparison_df: DataFrame a guardar.
        filename: Nombre del archivo CSV.
        folder: Carpeta relativa al proyecto.
    
    Returns:
        Path donde se guardó el archivo.
    """
    root = get_project_root()
    save_dir = root / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    comparison_df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"  ✓ Tabla guardada: {filepath}")
    
    return filepath
