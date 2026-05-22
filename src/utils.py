# -*- coding: utf-8 -*-
"""
Módulo de utilidades generales para el proyecto Fashion-MNIST.

Contiene funciones auxiliares para manejo de directorios, semillas aleatorias,
formateo de tiempo y otras utilidades comunes del proyecto.
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Optional, Union


# ============================================================================
# Constantes globales
# ============================================================================
RANDOM_STATE = 42


def get_project_root() -> Path:
    """
    Obtiene la ruta raíz del proyecto.
    
    Busca la raíz del proyecto navegando hacia arriba desde este archivo
    hasta encontrar el directorio que contiene 'data/fashion'.
    
    Returns:
        Path: Ruta absoluta al directorio raíz del proyecto.
    """
    # Este archivo está en src/utils.py, por lo que la raíz es un nivel arriba
    current = Path(__file__).resolve().parent.parent
    
    # Verificar que es la raíz correcta
    if (current / 'data' / 'fashion').exists():
        return current
    
    # Alternativa: buscar hacia arriba
    for parent in Path(__file__).resolve().parents:
        if (parent / 'data' / 'fashion').exists():
            return parent
    
    # Si no se encuentra, usar el directorio padre de src/
    return Path(__file__).resolve().parent.parent


def ensure_directories() -> dict:
    """
    Crea todos los directorios necesarios para el proyecto.
    
    Crea las carpetas de salida si no existen:
    - outputs/figures/
    - outputs/tables/
    - outputs/models/
    - data/processed/
    - notebooks/
    
    Returns:
        dict: Diccionario con las rutas creadas.
    """
    root = get_project_root()
    
    dirs = {
        'figures': root / 'outputs' / 'figures',
        'tables': root / 'outputs' / 'tables',
        'models': root / 'outputs' / 'models',
        'processed': root / 'data' / 'processed',
        'notebooks': root / 'notebooks',
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directorios del proyecto creados/verificados:")
    for name, path in dirs.items():
        print(f"  → {name}: {path}")
    
    return dirs


def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """
    Establece la semilla aleatoria para reproducibilidad.
    
    Configura la semilla en numpy y random para asegurar que los
    resultados sean reproducibles en todas las ejecuciones.
    
    Args:
        seed: Valor de la semilla aleatoria (default: 42).
    """
    np.random.seed(seed)
    random.seed(seed)
    print(f"✓ Semilla aleatoria establecida: {seed}")


def format_time(seconds: float) -> str:
    """
    Formatea un tiempo en segundos a una cadena legible.
    
    Args:
        seconds: Tiempo en segundos.
        
    Returns:
        str: Tiempo formateado (ej: '2m 30.5s', '45.3s', '0.123s').
    """
    if seconds < 0.001:
        return f"{seconds*1000:.3f} ms"
    elif seconds < 1:
        return f"{seconds:.3f} s"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def print_section_header(title: str, char: str = '=', width: int = 70) -> None:
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title: Título de la sección.
        char: Carácter para la línea decorativa.
        width: Ancho total del encabezado.
    """
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def save_figure(fig, filename: str, folder: str = 'outputs/figures', 
                dpi: int = 150, close: bool = True) -> Path:
    """
    Guarda una figura de matplotlib en el directorio especificado.
    
    Args:
        fig: Figura de matplotlib a guardar.
        filename: Nombre del archivo (sin ruta).
        folder: Carpeta relativa al proyecto donde guardar.
        dpi: Resolución de la imagen.
        close: Si True, cierra la figura después de guardar.
        
    Returns:
        Path: Ruta completa donde se guardó la figura.
    """
    import matplotlib.pyplot as plt
    
    root = get_project_root()
    save_dir = root / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    
    if close:
        plt.close(fig)
    
    print(f"  ✓ Figura guardada: {filepath}")
    return filepath


def save_dataframe(df, filename: str, folder: str = 'outputs/tables') -> Path:
    """
    Guarda un DataFrame como CSV en el directorio especificado.
    
    Args:
        df: DataFrame de pandas a guardar.
        filename: Nombre del archivo CSV (sin ruta).
        folder: Carpeta relativa al proyecto donde guardar.
        
    Returns:
        Path: Ruta completa donde se guardó el archivo.
    """
    root = get_project_root()
    save_dir = root / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"  ✓ Tabla guardada: {filepath}")
    return filepath
