# -*- coding: utf-8 -*-
"""
Módulo de modelos de Machine Learning - Proyecto Fashion-MNIST.

Funciones para crear, entrenar, evaluar y comparar modelos clásicos
de clasificación. Incluye pipelines con y sin PCA, validación cruzada
y búsqueda de hiperparámetros.
"""

import time
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (StratifiedKFold, cross_validate, 
                                      GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, classification_report)

warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def get_base_models() -> Dict[str, Any]:
    """
    Retorna un diccionario con los 9 modelos base requeridos.
    
    Los modelos incluyen:
    1. Perceptrón
    2. Adaline (SGD con pérdida cuadrática)
    3. Regresión Logística
    4. SVM Lineal
    5. SVM Polinómico
    6. SVM RBF
    7. KNN
    8. Árbol de Decisión
    9. Bosque Aleatorio
    
    Returns:
        Dict con nombre_modelo -> instancia del modelo.
    """
    models = {
        'Perceptron': Perceptron(
            max_iter=1000, random_state=RANDOM_STATE, tol=1e-3
        ),
        'Adaline (SGD)': SGDClassifier(
            loss='squared_error', max_iter=1000, 
            random_state=RANDOM_STATE, tol=1e-3
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, 
            solver='lbfgs'
        ),
        'SVM Linear': LinearSVC(
            max_iter=5000, random_state=RANDOM_STATE, dual='auto'
        ),
        'SVM Polynomial': SVC(
            kernel='poly', degree=3, random_state=RANDOM_STATE, probability=True
        ),
        'SVM RBF': SVC(
            kernel='rbf', random_state=RANDOM_STATE, probability=True
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    return models


def create_pipeline(model, use_pca: bool = False, 
                    n_components: float = 0.95) -> Pipeline:
    """
    Crea un pipeline de sklearn con escalado y clasificador.
    
    Pipeline A (sin PCA): StandardScaler → Clasificador
    Pipeline B (con PCA): StandardScaler → PCA → Clasificador
    
    Args:
        model: Instancia del clasificador.
        use_pca: Si True, incluye PCA en el pipeline.
        n_components: Número de componentes PCA o varianza explicada.
    
    Returns:
        Pipeline de sklearn configurado.
    """
    steps = [('scaler', StandardScaler())]
    
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)))
    
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def train_and_evaluate(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = '') -> Dict[str, Any]:
    """
    Entrena un pipeline y evalúa su rendimiento.
    
    Mide tiempo de entrenamiento y predicción, calcula métricas
    de clasificación (accuracy, precision, recall, F1).
    
    Args:
        pipeline: Pipeline de sklearn a entrenar.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        model_name: Nombre del modelo para reportes.
    
    Returns:
        Diccionario con métricas de rendimiento.
    """
    result = {'model_name': model_name}
    
    try:
        # Entrenar
        t_start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t_start
        result['train_time'] = train_time
        
        # Predecir
        t_start = time.time()
        y_pred = pipeline.predict(X_test)
        predict_time = time.time() - t_start
        result['predict_time'] = predict_time
        
        # Métricas
        result['accuracy'] = float(accuracy_score(y_test, y_pred))
        result['precision_macro'] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        result['recall_macro'] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        result['f1_macro'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        result['precision_weighted'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        result['recall_weighted'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        result['f1_weighted'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        result['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
        result['y_pred'] = y_pred
        result['pipeline'] = pipeline
        result['success'] = True
        
        print(f"  ✓ {model_name}: Accuracy={result['accuracy']:.4f}, "
              f"F1-macro={result['f1_macro']:.4f}, "
              f"Tiempo={train_time:.2f}s")
        
    except Exception as e:
        print(f"  ✗ Error en {model_name}: {e}")
        result['success'] = False
        result['error'] = str(e)
        result['accuracy'] = 0.0
        result['f1_macro'] = 0.0
        result['train_time'] = 0.0
        result['predict_time'] = 0.0
    
    return result


def train_all_models(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     use_pca: bool = False, 
                     n_components: float = 0.95) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Entrena y evalúa todos los 9 modelos base.
    
    Args:
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de prueba.
        use_pca: Si incluir PCA en los pipelines.
        n_components: Componentes PCA.
    
    Returns:
        Tuple con (lista de resultados, DataFrame de comparación).
    """
    pca_label = "con PCA" if use_pca else "sin PCA"
    print(f"\n{'='*60}")
    print(f"  Entrenando todos los modelos ({pca_label})")
    print(f"{'='*60}")
    
    models = get_base_models()
    results = []
    
    for name, model in models.items():
        pipeline = create_pipeline(model, use_pca=use_pca, n_components=n_components)
        result = train_and_evaluate(pipeline, X_train, y_train, X_test, y_test, name)
        results.append(result)
    
    # Crear tabla comparativa
    comparison_data = []
    for r in results:
        if r.get('success', False):
            comparison_data.append({
                'Modelo': r['model_name'],
                'Accuracy': r['accuracy'],
                'Precision_macro': r.get('precision_macro', 0),
                'Recall_macro': r.get('recall_macro', 0),
                'F1_macro': r['f1_macro'],
                'F1_weighted': r.get('f1_weighted', 0),
                'Tiempo_entrenamiento': r['train_time'],
                'Tiempo_prediccion': r['predict_time'],
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('F1_macro', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"  Resumen - {pca_label}")
    print(f"{'='*60}")
    if len(comparison_df) > 0:
        print(comparison_df[['Modelo', 'Accuracy', 'F1_macro', 'Tiempo_entrenamiento']].to_string(index=False))
    
    return results, comparison_df


def get_hyperparameter_grids() -> Dict[str, Dict]:
    """
    Retorna los grids de hiperparámetros para búsqueda.
    
    Los nombres de parámetros usan el prefijo 'classifier__'
    para funcionar con los Pipelines de sklearn.
    
    Returns:
        Dict con nombre_modelo -> param_grid.
    """
    grids = {
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2'],
        },
        'SVM Linear': {
            'classifier__C': [0.01, 0.1, 1, 10],
        },
        'SVM RBF': {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 0.01, 0.1],
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan'],
        },
        'Decision Tree': {
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 5],
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
        },
    }
    return grids


def hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray,
                          model_name: str, pipeline: Pipeline,
                          param_grid: Dict, cv: int = 5,
                          scoring: str = 'f1_macro',
                          n_iter: Optional[int] = None) -> Tuple[Any, Dict, Any]:
    """
    Realiza búsqueda de hiperparámetros con validación cruzada.
    
    Usa GridSearchCV si n_iter es None, o RandomizedSearchCV si se especifica.
    
    Args:
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        model_name: Nombre del modelo.
        pipeline: Pipeline de sklearn.
        param_grid: Diccionario de hiperparámetros a buscar.
        cv: Número de folds para validación cruzada.
        scoring: Métrica de evaluación.
        n_iter: Si se especifica, usa RandomizedSearchCV con n iteraciones.
    
    Returns:
        Tuple con (mejor_estimador, mejores_params, resultados_cv).
    """
    print(f"\n  Búsqueda de hiperparámetros: {model_name}")
    print(f"    Grid: {param_grid}")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        if n_iter is not None:
            search = RandomizedSearchCV(
                pipeline, param_grid, n_iter=n_iter,
                cv=skf, scoring=scoring, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=0
            )
        else:
            search = GridSearchCV(
                pipeline, param_grid, cv=skf,
                scoring=scoring, n_jobs=-1, verbose=0
            )
        
        t_start = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - t_start
        
        print(f"    ✓ Mejor score: {search.best_score_:.4f}")
        print(f"    ✓ Mejores params: {search.best_params_}")
        print(f"    ✓ Tiempo: {search_time:.1f}s")
        
        return search.best_estimator_, search.best_params_, search.cv_results_
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return pipeline, {}, {}


def cross_validate_models(X_train: np.ndarray, y_train: np.ndarray,
                          models_dict: Optional[Dict] = None,
                          cv: int = 5, use_pca: bool = False,
                          n_components: float = 0.95) -> pd.DataFrame:
    """
    Realiza validación cruzada estratificada para todos los modelos.
    
    Args:
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        models_dict: Diccionario de modelos. Si None, usa get_base_models().
        cv: Número de folds.
        use_pca: Si incluir PCA.
        n_components: Componentes PCA.
    
    Returns:
        DataFrame con resultados de validación cruzada por modelo.
    """
    if models_dict is None:
        models_dict = get_base_models()
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    pca_label = "con PCA" if use_pca else "sin PCA"
    print(f"\n{'='*60}")
    print(f"  Validación Cruzada ({cv} folds, {pca_label})")
    print(f"{'='*60}")
    
    cv_results = []
    
    for name, model in models_dict.items():
        pipeline = create_pipeline(model, use_pca=use_pca, n_components=n_components)
        
        try:
            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            cv_result = cross_validate(
                pipeline, X_train, y_train, cv=skf,
                scoring=scoring, n_jobs=-1, return_train_score=True
            )
            
            result = {
                'Modelo': name,
                'CV_Accuracy_mean': np.mean(cv_result['test_accuracy']),
                'CV_Accuracy_std': np.std(cv_result['test_accuracy']),
                'CV_F1_macro_mean': np.mean(cv_result['test_f1_macro']),
                'CV_F1_macro_std': np.std(cv_result['test_f1_macro']),
                'CV_Precision_mean': np.mean(cv_result['test_precision_macro']),
                'CV_Recall_mean': np.mean(cv_result['test_recall_macro']),
                'Train_Accuracy_mean': np.mean(cv_result['train_accuracy']),
                'Overfit_gap': np.mean(cv_result['train_accuracy']) - np.mean(cv_result['test_accuracy']),
            }
            
            cv_results.append(result)
            
            print(f"  ✓ {name}: CV-Accuracy={result['CV_Accuracy_mean']:.4f} "
                  f"(±{result['CV_Accuracy_std']:.4f}), "
                  f"CV-F1={result['CV_F1_macro_mean']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error en {name}: {e}")
            cv_results.append({
                'Modelo': name,
                'CV_Accuracy_mean': 0.0,
                'CV_Accuracy_std': 0.0,
                'CV_F1_macro_mean': 0.0,
                'CV_F1_macro_std': 0.0,
                'CV_Precision_mean': 0.0,
                'CV_Recall_mean': 0.0,
                'Train_Accuracy_mean': 0.0,
                'Overfit_gap': 0.0,
            })
    
    df = pd.DataFrame(cv_results)
    df = df.sort_values('CV_F1_macro_mean', ascending=False)
    
    return df
