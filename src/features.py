# -*- coding: utf-8 -*-
"""
Módulo de características - Proyecto Final Fashion-MNIST.

Implementa la extracción de características visuales inspirada en
los métodos desarrollados durante el curso (HOG, Proyecciones, Densidad,
Transiciones y Bounding Box).

Estas características convierten una imagen 28x28 (784 píxeles) 
en un vector de 205 características numéricas con alto poder discriminativo.
"""

import numpy as np
import pandas as pd
from skimage.feature import hog

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def get_feature_names():
    """Genera los nombres de las 205 características."""
    names = []
    # 1. HOG (144)
    names.extend([f"hog_{i}" for i in range(144)])
    # 2. Proyección horizontal (28)
    names.extend([f"proj_h_{i}" for i in range(28)])
    # 3. Proyección vertical (28)
    names.extend([f"proj_v_{i}" for i in range(28)])
    # 4. Densidad (1)
    names.append("densidad_pixels")
    # 5. Ratio superior/inferior (1)
    names.append("ratio_superior_inferior")
    # 6. Desviación estándar de la proyección horizontal (1)
    names.append("std_proj_h")
    # 7. Relación alto/ancho del bounding box (1)
    names.append("bbox_ratio")
    # 8. Promedio de transiciones horizontales (1)
    names.append("transiciones_h_promedio")
    
    return names


def extraer_caracteristicas(imagenes, umbral=30):
    """
    Extrae el conjunto de 205 características de un arreglo de imágenes.
    
    Args:
        imagenes: Array de numpy con shape (N, 784) o (N, 28, 28). Rango [0, 255].
        umbral: Valor de intensidad para binarizar la imagen.
        
    Returns:
        DataFrame de pandas con las características extraídas.
    """
    features_list = []
    
    # Asegurar que las imágenes son floats para cálculos numéricos
    # y hacer reshape si vienen planas
    if len(imagenes.shape) == 2:
        num_imgs = imagenes.shape[0]
        imgs_2d = imagenes.reshape(num_imgs, 28, 28).astype(np.float32)
    else:
        imgs_2d = imagenes.astype(np.float32)

    print(f"  Extrayendo características para {len(imgs_2d)} imágenes...")
    
    for img in tqdm(imgs_2d, desc="Feature Extraction"):
        # Asegurarse de que los valores no sean cero para divisiones seguras
        img_safe = img + 1e-6
        bin_ = (img > umbral).astype(np.float32)
        
        # 1. HOG: 4x4 celdas x 9 orientaciones = 144 valores
        # Usamos block_norm='L2-Hys' que es el estándar, y feature_vector=True
        hog_feat = hog(img, orientations=9, pixels_per_cell=(7, 7),
                       cells_per_block=(1, 1), visualize=False, feature_vector=True)
        
        # 2 y 3. Proyección horizontal y vertical (normalizadas por 255)
        # Esto captura el "perfil" del objeto en la imagen
        proj_h = img.sum(axis=1) / 255.0
        proj_v = img.sum(axis=0) / 255.0
        
        # 4. Densidad de pixels activos
        densidad = np.array([bin_.sum() / (28 * 28)])
        
        # 5. Ratio tinta superior / tinta inferior
        # Útil para distinguir pantalones (tinta uniforme) de camisetas (más tinta arriba)
        suma_superior = img[:14, :].sum()
        suma_inferior = img[14:, :].sum() + 1e-6
        ratio_ud = np.array([suma_superior / suma_inferior])
        
        # 6. Std de la proyección horizontal
        # Indica qué tan variable es el ancho del objeto a lo largo de su altura
        std_projh = np.array([img.sum(axis=1).std()])
        
        # 7. Relación alto/ancho del bounding box
        filas_act = np.where(img.sum(axis=1) > 0)[0]
        cols_act  = np.where(img.sum(axis=0) > 0)[0]
        
        alto  = (filas_act[-1] - filas_act[0] + 1) if len(filas_act) > 1 else 1
        ancho = (cols_act[-1]  - cols_act[0]  + 1) if len(cols_act)  > 1 else 1
        bbox_ratio = np.array([alto / ancho])
        
        # 8. Promedio de transiciones horizontales por fila
        # Captura la textura y bordes internos. Cuántas veces pasa de negro a blanco y viceversa.
        trans_h = np.diff((img > umbral).astype(int), axis=1)
        n_trans = np.array([np.sum(trans_h != 0, axis=1).mean()])
        
        # Concatenar todas las características en un vector de 205 valores
        vector = np.concatenate([
            hog_feat, proj_h, proj_v, densidad, ratio_ud, std_projh, bbox_ratio, n_trans
        ])
        features_list.append(vector)

    # Convertir a DataFrame
    nombres_columnas = get_feature_names()
    df_features = pd.DataFrame(features_list, columns=nombres_columnas)
    
    return df_features

def get_feature_documentation():
    """Retorna un DataFrame con la documentación de cada característica."""
    docs = [
        {"Característica": "hog_0 a hog_143", "Familia": "HOG (Forma y Textura)", "Descripción": "Histogram of Oriented Gradients. Captura la dirección de los bordes y texturas en celdas de 7x7 píxeles."},
        {"Característica": "proj_h_0 a proj_h_27", "Familia": "Perfil (Intensidad)", "Descripción": "Proyección horizontal de la imagen. Suma de intensidades por fila (normalizada)."},
        {"Característica": "proj_v_0 a proj_v_27", "Familia": "Perfil (Intensidad)", "Descripción": "Proyección vertical de la imagen. Suma de intensidades por columna (normalizada)."},
        {"Característica": "densidad_pixels", "Familia": "Global", "Descripción": "Proporción de píxeles activos (mayores al umbral) sobre el total del área (28x28)."},
        {"Característica": "ratio_superior_inferior", "Familia": "Global", "Descripción": "Relación entre la cantidad de tinta en la mitad superior y la mitad inferior de la imagen."},
        {"Característica": "std_proj_h", "Familia": "Perfil (Intensidad)", "Descripción": "Desviación estándar de la proyección horizontal. Mide la variabilidad del ancho a lo largo del alto."},
        {"Característica": "bbox_ratio", "Familia": "Forma", "Descripción": "Relación entre el alto y el ancho del bounding box del objeto (caja delimitadora)."},
        {"Característica": "transiciones_h_promedio", "Familia": "Textura y Bordes", "Descripción": "Número promedio de veces que el color cambia de blanco a negro (o viceversa) por fila."}
    ]
    return pd.DataFrame(docs)

