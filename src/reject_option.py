# -*- coding: utf-8 -*-
"""
Módulo de opción de rechazo - Proyecto Fashion-MNIST.

Implementa la opción de rechazo para modelos de clasificación.
Cuando el modelo no está suficientemente seguro de su predicción
(probabilidad máxima < umbral), la predicción se rechaza.

Esto es especialmente útil para clases visualmente similares
como T-shirt/top y Shirt.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from .utils import get_project_root


def get_prediction_probabilities(model, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Obtiene probabilidades de predicción del modelo.
    
    Intenta usar predict_proba primero, luego decision_function.
    
    Args:
        model: Modelo o pipeline entrenado.
        X: Datos de entrada.
    
    Returns:
        Array de probabilidades, shape (n, n_clases), o None si falla.
    """
    try:
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        elif hasattr(model, 'decision_function'):
            decisions = model.decision_function(X)
            # Convertir decision_function a probabilidades con softmax
            if decisions.ndim == 1:
                decisions = decisions.reshape(-1, 1)
            from scipy.special import softmax
            return softmax(decisions, axis=1)
    except Exception as e:
        print(f"  ⚠ No se pudieron obtener probabilidades: {e}")
    
    return None


def ensure_probability_model(model, X_train: np.ndarray, 
                              y_train: np.ndarray) -> Any:
    """
    Asegura que el modelo pueda generar probabilidades.
    
    Si el modelo no tiene predict_proba, lo envuelve con
    CalibratedClassifierCV para calibrar las probabilidades.
    
    Args:
        model: Modelo entrenado.
        X_train: Datos de entrenamiento para calibración.
        y_train: Etiquetas de entrenamiento.
    
    Returns:
        Modelo con capacidad predict_proba.
    """
    if hasattr(model, 'predict_proba'):
        try:
            # Verificar que realmente funciona
            model.predict_proba(X_train[:2])
            return model
        except Exception:
            pass
    
    print("  ℹ Calibrando modelo con CalibratedClassifierCV...")
    
    try:
        calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        calibrated.fit(X_train, y_train)
        print("  ✓ Modelo calibrado exitosamente.")
        return calibrated
    except Exception as e:
        print(f"  ⚠ Error en calibración: {e}")
        return model


def apply_reject_option(probas: np.ndarray, threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica opción de rechazo basada en umbral de confianza.
    
    Si la probabilidad máxima de una predicción es menor al umbral,
    la predicción se marca como "rechazada".
    
    Args:
        probas: Array de probabilidades, shape (n, n_clases).
        threshold: Umbral mínimo de confianza.
    
    Returns:
        Tuple con:
            - predictions: Clase predicha (argmax), -1 para rechazados.
            - rejected_mask: Array booleano indicando muestras rechazadas.
    """
    max_probas = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    
    rejected_mask = max_probas < threshold
    predictions_with_reject = predictions.copy()
    predictions_with_reject[rejected_mask] = -1
    
    return predictions_with_reject, rejected_mask


def evaluate_with_rejection(y_true: np.ndarray, y_pred_original: np.ndarray,
                             probas: np.ndarray,
                             thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                             class_names: Optional[dict] = None) -> pd.DataFrame:
    """
    Evalúa el modelo con diferentes umbrales de rechazo.
    
    Para cada umbral calcula: tasa de rechazo, accuracy sobre
    muestras aceptadas, errores evitados, etc.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred_original: Predicciones originales (sin rechazo).
        probas: Probabilidades de predicción.
        thresholds: Lista de umbrales a evaluar.
        class_names: Diccionario {id: nombre}.
    
    Returns:
        DataFrame con resultados por umbral.
    """
    original_accuracy = accuracy_score(y_true, y_pred_original)
    original_errors = np.sum(y_true != y_pred_original)
    
    results = []
    
    for threshold in thresholds:
        max_probas = probas.max(axis=1)
        rejected_mask = max_probas < threshold
        accepted_mask = ~rejected_mask
        
        n_total = len(y_true)
        n_rejected = rejected_mask.sum()
        n_accepted = accepted_mask.sum()
        rejection_rate = n_rejected / n_total
        
        # Accuracy sobre muestras aceptadas
        if n_accepted > 0:
            accuracy_accepted = accuracy_score(y_true[accepted_mask], 
                                                y_pred_original[accepted_mask])
        else:
            accuracy_accepted = 0.0
        
        # Errores evitados (errores que caen en rechazados)
        errors_in_rejected = np.sum(
            (y_true != y_pred_original) & rejected_mask
        )
        
        # Clase más rechazada
        if n_rejected > 0 and class_names is not None:
            rejected_classes = y_true[rejected_mask]
            most_rejected_class_id = np.bincount(rejected_classes.astype(int)).argmax()
            most_rejected_class = class_names.get(most_rejected_class_id, 
                                                    str(most_rejected_class_id))
        else:
            most_rejected_class = 'N/A'
        
        results.append({
            'Umbral': threshold,
            'N_aceptadas': n_accepted,
            'N_rechazadas': n_rejected,
            'Tasa_rechazo': round(rejection_rate, 4),
            'Accuracy_original': round(original_accuracy, 4),
            'Accuracy_aceptadas': round(accuracy_accepted, 4),
            'Mejora_accuracy': round(accuracy_accepted - original_accuracy, 4),
            'Errores_evitados': errors_in_rejected,
            'Errores_originales': original_errors,
            'Clase_más_rechazada': most_rejected_class,
        })
    
    return pd.DataFrame(results)


def plot_reject_option_analysis(reject_results_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Genera gráfica de accuracy vs tasa de rechazo a diferentes umbrales.
    
    Usa doble eje Y: accuracy a la izquierda, tasa de rechazo a la derecha.
    
    Args:
        reject_results_df: DataFrame de evaluate_with_rejection.
        save_path: Ruta donde guardar.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = reject_results_df['Umbral']
    
    # Eje izquierdo: Accuracy
    color_acc = '#2196F3'
    ax1.plot(x, reject_results_df['Accuracy_aceptadas'], 'o-', 
             color=color_acc, linewidth=2, markersize=8, label='Accuracy (aceptadas)')
    ax1.axhline(y=reject_results_df['Accuracy_original'].iloc[0], 
                color=color_acc, linestyle='--', alpha=0.5, label='Accuracy (sin rechazo)')
    ax1.set_xlabel('Umbral de Confianza', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color=color_acc)
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0.5, 1.05)
    
    # Eje derecho: Tasa de rechazo
    ax2 = ax1.twinx()
    color_rej = '#F44336'
    ax2.plot(x, reject_results_df['Tasa_rechazo'] * 100, 's--', 
             color=color_rej, linewidth=2, markersize=8, label='Tasa de rechazo (%)')
    ax2.set_ylabel('Tasa de Rechazo (%)', fontsize=12, color=color_rej)
    ax2.tick_params(axis='y', labelcolor=color_rej)
    ax2.set_ylim(0, 100)
    
    # Combinar leyendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10)
    
    ax1.set_title('Análisis de Opción de Rechazo\nAccuracy vs Tasa de Rechazo',
                   fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Análisis de rechazo guardado: {save_path}")
        plt.close(fig)
    
    return fig


def plot_rejected_samples(X_test_images: np.ndarray, y_true: np.ndarray,
                           probas: np.ndarray, threshold: float,
                           class_names: dict, n: int = 10,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Muestra imágenes que serían rechazadas con su confianza.
    
    Args:
        X_test_images: Imágenes originales, shape (n, 784).
        y_true: Etiquetas verdaderas.
        probas: Probabilidades de predicción.
        threshold: Umbral de rechazo.
        class_names: Diccionario {id: nombre}.
        n: Número de imágenes a mostrar.
        save_path: Ruta donde guardar.
    """
    max_probas = probas.max(axis=1)
    rejected_mask = max_probas < threshold
    rejected_idx = np.where(rejected_mask)[0]
    
    if len(rejected_idx) == 0:
        print(f"  ℹ No hay muestras rechazadas con umbral {threshold}")
        return None
    
    sample_size = min(n, len(rejected_idx))
    # Seleccionar las más inciertas
    uncertainties = max_probas[rejected_idx]
    most_uncertain = np.argsort(uncertainties)[:sample_size]
    sample_idx = rejected_idx[most_uncertain]
    
    n_cols = 5
    n_rows = (sample_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
    fig.suptitle(f'Muestras Rechazadas (umbral = {threshold})', 
                 fontsize=14, fontweight='bold', color='orange')
    
    axes_flat = axes.flatten() if n_rows > 1 else (axes if isinstance(axes, np.ndarray) else [axes])
    
    for i, idx in enumerate(sample_idx):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        ax.imshow(X_test_images[idx].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
        
        true_name = class_names.get(y_true[idx], str(y_true[idx]))
        pred_class = probas[idx].argmax()
        pred_name = class_names.get(pred_class, str(pred_class))
        conf = max_probas[idx]
        
        ax.set_title(f'Real: {true_name}\nPred: {pred_name}\nConf: {conf:.3f}',
                      fontsize=8, color='orange', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    for i in range(sample_size, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Muestras rechazadas guardadas: {save_path}")
        plt.close(fig)
    
    return fig


def analyze_rejected_by_class(y_true: np.ndarray, probas: np.ndarray,
                                threshold: float, 
                                class_names: dict) -> pd.DataFrame:
    """
    Analiza la tasa de rechazo por clase.
    
    Args:
        y_true: Etiquetas verdaderas.
        probas: Probabilidades de predicción.
        threshold: Umbral de rechazo.
        class_names: Diccionario {id: nombre}.
    
    Returns:
        DataFrame con tasa de rechazo por clase.
    """
    max_probas = probas.max(axis=1)
    rejected_mask = max_probas < threshold
    
    results = []
    for cls_id in sorted(class_names.keys()):
        cls_mask = y_true == cls_id
        n_total = cls_mask.sum()
        n_rejected = (cls_mask & rejected_mask).sum()
        
        results.append({
            'Clase': cls_id,
            'Nombre': class_names[cls_id],
            'Total': n_total,
            'Rechazadas': n_rejected,
            'Tasa_rechazo': round(n_rejected / n_total, 4) if n_total > 0 else 0,
        })
    
    return pd.DataFrame(results)


def full_reject_analysis(model, X_test: np.ndarray, y_test: np.ndarray,
                          X_test_images: np.ndarray, class_names: dict,
                          X_train: Optional[np.ndarray] = None,
                          y_train: Optional[np.ndarray] = None,
                          thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                          save_dir: str = 'outputs/figures') -> Dict[str, Any]:
    """
    Análisis completo de opción de rechazo.
    
    Obtiene probabilidades, evalúa múltiples umbrales, genera gráficas
    y tablas de análisis.
    
    Args:
        model: Modelo o pipeline entrenado.
        X_test: Características de prueba (tabulares).
        y_test: Etiquetas de prueba.
        X_test_images: Imágenes originales de prueba.
        class_names: Diccionario {id: nombre}.
        X_train: Datos de entrenamiento (para calibración si necesario).
        y_train: Etiquetas de entrenamiento (para calibración).
        thresholds: Umbrales a evaluar.
        save_dir: Directorio para guardar figuras.
    
    Returns:
        Diccionario con resultados del análisis.
    """
    root = get_project_root()
    save_path = root / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Obtener probabilidades
    probas = get_prediction_probabilities(model, X_test)
    
    if probas is None and X_train is not None and y_train is not None:
        print("  Intentando calibrar modelo para obtener probabilidades...")
        calibrated_model = ensure_probability_model(model, X_train, y_train)
        probas = get_prediction_probabilities(calibrated_model, X_test)
    
    if probas is None:
        print("  ✗ No se pudieron obtener probabilidades para opción de rechazo.")
        return results
    
    # Predicciones originales
    y_pred = model.predict(X_test)
    
    # Mapear clases del modelo a las clases reales si necesario
    labels = sorted(class_names.keys())
    
    # Evaluar con diferentes umbrales
    reject_df = evaluate_with_rejection(y_test, y_pred, probas, thresholds, class_names)
    results['reject_results'] = reject_df
    
    print("\n  Resultados de opción de rechazo:")
    print(reject_df.to_string(index=False))
    
    # Gráfica de análisis
    plot_reject_option_analysis(reject_df, 
                                save_path=str(save_path / 'reject_option_analysis.png'))
    
    # Muestras rechazadas con umbral 0.7
    plot_rejected_samples(X_test_images, y_test, probas, 0.7, class_names,
                           save_path=str(save_path / 'rejected_samples.png'))
    
    # Análisis por clase con umbral 0.7
    class_reject_df = analyze_rejected_by_class(y_test, probas, 0.7, class_names)
    results['class_reject'] = class_reject_df
    
    print("\n  Tasa de rechazo por clase (umbral=0.7):")
    print(class_reject_df.to_string(index=False))
    
    # Guardar tabla
    tables_dir = root / 'outputs' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    reject_df.to_csv(tables_dir / 'reject_option_results.csv', index=False)
    class_reject_df.to_csv(tables_dir / 'reject_by_class.csv', index=False)
    
    results['probas'] = probas
    
    return results
