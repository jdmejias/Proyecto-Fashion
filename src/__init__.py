# -*- coding: utf-8 -*-
"""
Paquete src - Proyecto Final Fashion-MNIST
Clasificación de imágenes con Machine Learning Clásico

Este paquete contiene los módulos necesarios para el pipeline completo
de clasificación de imágenes Fashion-MNIST usando extracción manual
de características visuales y modelos clásicos de ML.
"""

from .data_loader import (
    load_fashion_mnist,
    filter_classes,
    get_class_names,
    create_stratified_sample,
    RANDOM_STATE,
    SELECTED_CLASSES,
    CLASS_NAMES,
    IMG_SIZE
)

from .utils import (
    ensure_directories,
    get_project_root,
    set_random_seed,
    format_time,
    print_section_header
)

__version__ = '1.0.0'
__author__ = 'Proyecto Final ML'
