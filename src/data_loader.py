# -*- coding: utf-8 -*-
"""
Módulo de carga de datos - Proyecto Fashion-MNIST.

Funciones para cargar el dataset Fashion-MNIST desde archivos IDX comprimidos,
filtrar clases seleccionadas y crear muestras estratificadas.
"""

import os
import gzip
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from .utils import get_project_root


# ============================================================================
# Constantes del proyecto
# ============================================================================
RANDOM_STATE = 42
IMG_SIZE = 28
SELECTED_CLASSES = [0, 1, 3, 6, 8]

CLASS_NAMES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

SELECTED_CLASS_NAMES = {k: CLASS_NAMES[k] for k in SELECTED_CLASSES}

# Nombres en español para las clases seleccionadas
CLASS_NAMES_ES = {
    0: 'Camiseta',
    1: 'Pantalón',
    3: 'Vestido',
    6: 'Camisa',
    8: 'Bolso'
}


def get_class_names(selected_only: bool = True) -> Dict[int, str]:
    """
    Retorna un diccionario con los nombres de las clases.
    
    Args:
        selected_only: Si True, retorna solo las clases seleccionadas.
        
    Returns:
        Dict con id_clase -> nombre_clase.
    """
    if selected_only:
        return SELECTED_CLASS_NAMES.copy()
    return CLASS_NAMES.copy()


def _load_idx_file(filepath: str) -> np.ndarray:
    """
    Lee un archivo IDX comprimido con gzip.
    
    Args:
        filepath: Ruta al archivo .gz.
        
    Returns:
        np.ndarray con los datos leídos.
    """
    with gzip.open(filepath, 'rb') as f:
        data = f.read()
    
    # Leer el magic number para determinar el tipo
    magic = int.from_bytes(data[0:4], byteorder='big')
    
    # Los primeros 4 bytes son magic, los siguientes son dimensiones
    if magic == 2051:  # Imágenes
        n_items = int.from_bytes(data[4:8], byteorder='big')
        n_rows = int.from_bytes(data[8:12], byteorder='big')
        n_cols = int.from_bytes(data[12:16], byteorder='big')
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(n_items, n_rows * n_cols)
        return images
    elif magic == 2049:  # Etiquetas
        n_items = int.from_bytes(data[4:8], byteorder='big')
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels
    else:
        raise ValueError(f"Magic number no reconocido: {magic}")


def load_fashion_mnist(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset Fashion-MNIST completo desde archivos IDX comprimidos.
    
    Lee los archivos .gz del directorio data/fashion/ del proyecto y retorna
    los conjuntos de entrenamiento y prueba.
    
    Args:
        data_path: Ruta al directorio con los archivos .gz.
                   Si es None, usa data/fashion/ desde la raíz del proyecto.
    
    Returns:
        Tuple con (X_train, y_train, X_test, y_test):
            - X_train: Imágenes de entrenamiento, shape (60000, 784), dtype uint8.
            - y_train: Etiquetas de entrenamiento, shape (60000,), dtype uint8.
            - X_test: Imágenes de prueba, shape (10000, 784), dtype uint8.
            - y_test: Etiquetas de prueba, shape (10000,), dtype uint8.
    
    Raises:
        FileNotFoundError: Si no se encuentran los archivos del dataset.
    """
    if data_path is None:
        data_path = get_project_root() / 'data' / 'fashion'
    else:
        data_path = Path(data_path)
    
    # Verificar que el directorio existe
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de datos: {data_path}\n"
            f"Asegúrese de que el dataset Fashion-MNIST está en data/fashion/"
        )
    
    # Definir archivos
    files = {
        'train_images': data_path / 'train-images-idx3-ubyte.gz',
        'train_labels': data_path / 'train-labels-idx1-ubyte.gz',
        'test_images': data_path / 't10k-images-idx3-ubyte.gz',
        'test_labels': data_path / 't10k-labels-idx1-ubyte.gz',
    }
    
    # Verificar que todos los archivos existen
    for name, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    print("Cargando dataset Fashion-MNIST...")
    
    # Método 1: Usar gzip + numpy directamente
    try:
        with gzip.open(str(files['train_labels']), 'rb') as f:
            y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
        with gzip.open(str(files['train_images']), 'rb') as f:
            X_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(y_train), 784)
        
        with gzip.open(str(files['test_labels']), 'rb') as f:
            y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
        with gzip.open(str(files['test_images']), 'rb') as f:
            X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(y_test), 784)
            
    except Exception as e:
        print(f"  ⚠ Error con método directo, intentando método alternativo: {e}")
        X_train = _load_idx_file(str(files['train_images']))
        y_train = _load_idx_file(str(files['train_labels']))
        X_test = _load_idx_file(str(files['test_images']))
        y_test = _load_idx_file(str(files['test_labels']))
    
    print(f"  ✓ Datos de entrenamiento: {X_train.shape[0]:,} imágenes")
    print(f"  ✓ Datos de prueba: {X_test.shape[0]:,} imágenes")
    print(f"  ✓ Dimensión de cada imagen: {IMG_SIZE}x{IMG_SIZE} = {X_train.shape[1]} píxeles")
    print(f"  ✓ Tipo de dato: {X_train.dtype}")
    print(f"  ✓ Rango de valores: [{X_train.min()}, {X_train.max()}]")
    print(f"  ✓ Clases únicas: {np.unique(y_train)}")
    
    return X_train, y_train, X_test, y_test


def filter_classes(X: np.ndarray, y: np.ndarray, 
                   selected_classes: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra el dataset para mantener solo las clases seleccionadas.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas, shape (n,).
        selected_classes: Lista de IDs de clases a mantener.
                         Si es None, usa SELECTED_CLASSES.
    
    Returns:
        Tuple con (X_filtrado, y_filtrado).
    """
    if selected_classes is None:
        selected_classes = SELECTED_CLASSES
    
    # Crear máscara booleana
    mask = np.isin(y, selected_classes)
    X_filtered = X[mask].copy()
    y_filtered = y[mask].copy()
    
    class_names = get_class_names()
    print(f"\n✓ Dataset filtrado a {len(selected_classes)} clases:")
    for cls_id in selected_classes:
        n_samples = np.sum(y_filtered == cls_id)
        name = class_names.get(cls_id, f'Clase {cls_id}')
        print(f"  → Clase {cls_id} ({name}): {n_samples:,} imágenes")
    
    print(f"  → Total: {len(X_filtered):,} imágenes (de {len(X):,} originales)")
    
    return X_filtered, y_filtered


def create_stratified_sample(X: np.ndarray, y: np.ndarray, 
                             n_samples: int = 10000,
                             random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una muestra estratificada del dataset.
    
    Mantiene la proporción de clases en la muestra resultante.
    Útil para acelerar pruebas durante el desarrollo.
    
    Args:
        X: Array de imágenes, shape (n, 784).
        y: Array de etiquetas, shape (n,).
        n_samples: Número total de muestras deseadas.
        random_state: Semilla aleatoria.
    
    Returns:
        Tuple con (X_muestra, y_muestra).
    """
    from sklearn.model_selection import train_test_split
    
    if n_samples >= len(X):
        print(f"  ℹ n_samples ({n_samples}) >= total ({len(X)}), usando todos los datos.")
        return X.copy(), y.copy()
    
    # Calcular la proporción a mantener
    test_size = 1.0 - (n_samples / len(X))
    
    X_sample, _, y_sample, _ = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    class_names = get_class_names()
    print(f"\n✓ Muestra estratificada creada: {len(X_sample):,} imágenes")
    for cls_id in np.unique(y_sample):
        n = np.sum(y_sample == cls_id)
        name = class_names.get(cls_id, f'Clase {cls_id}')
        print(f"  → Clase {cls_id} ({name}): {n:,} ({100*n/len(y_sample):.1f}%)")
    
    return X_sample, y_sample
