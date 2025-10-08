
# ============================================
# core/validators.py
# ============================================
"""
Funciones de validación para parámetros e imágenes
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
import re


def validate_image(image: np.ndarray, 
                   min_size: Tuple[int, int] = (10, 10),
                   max_size: Tuple[int, int] = (5000, 5000)) -> bool:
    """
    Valida que una imagen sea válida y esté en el rango de tamaño correcto
    
    Args:
        image: Imagen a validar
        min_size: Tamaño mínimo (width, height)
        max_size: Tamaño máximo (width, height)
    
    Returns:
        True si la imagen es válida
    """
    if image is None:
        st.error("❌ La imagen es None")
        return False
    
    if not isinstance(image, np.ndarray):
        st.error("❌ La imagen debe ser un numpy array")
        return False
    
    if len(image.shape) not in [2, 3]:
        st.error("❌ La imagen debe tener 2 o 3 dimensiones")
        return False
    
    h, w = image.shape[:2]
    
    if w < min_size[0] or h < min_size[1]:
        st.error(f"❌ La imagen es demasiado pequeña. Mínimo: {min_size[0]}x{min_size[1]}px")
        return False
    
    if w > max_size[0] or h > max_size[1]:
        st.warning(f"⚠️ La imagen es muy grande. Será redimensionada automáticamente.")
        return True
    
    return True


def validate_kernel_size(size: int, must_be_odd: bool = True) -> int:
    """
    Valida y ajusta el tamaño de un kernel
    
    Args:
        size: Tamaño del kernel
        must_be_odd: Si debe ser impar
    
    Returns:
        Tamaño validado del kernel
    """
    if size < 1:
        st.warning("⚠️ El tamaño del kernel debe ser mayor a 0. Usando 3.")
        return 3
    
    if must_be_odd and size % 2 == 0:
        size += 1
        st.info(f"ℹ️ El tamaño del kernel debe ser impar. Ajustado a {size}.")
    
    return size


def validate_threshold_value(value: int, min_val: int = 0, max_val: int = 255) -> int:
    """
    Valida un valor de umbral
    
    Args:
        value: Valor a validar
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
    
    Returns:
        Valor validado
    """
    if value < min_val:
        st.warning(f"⚠️ Valor muy bajo. Usando mínimo {min_val}")
        return min_val
    
    if value > max_val:
        st.warning(f"⚠️ Valor muy alto. Usando máximo {max_val}")
        return max_val
    
    return value


def validate_file_path(file_path: str, must_exist: bool = True) -> Optional[Path]:
    """
    Valida una ruta de archivo
    
    Args:
        file_path: Ruta a validar
        must_exist: Si el archivo debe existir
    
    Returns:
        Path object si es válido, None en caso contrario
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            st.error(f"❌ El archivo no existe: {file_path}")
            return None
        
        if must_exist and not path.is_file():
            st.error(f"❌ La ruta no es un archivo válido: {file_path}")
            return None
        
        return path
        
    except Exception as e:
        st.error(f"❌ Error al validar ruta: {str(e)}")
        return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo removiendo caracteres no válidos
    
    Args:
        filename: Nombre de archivo a sanitizar
    
    Returns:
        Nombre de archivo sanitizado
    """
    # Remover caracteres no válidos
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limitar longitud
    max_length = 255
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:max_length - len(ext) - 1]
        filename = f"{name}.{ext}" if ext else name
    
    return filename


def validate_color_value(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Valida valores de color BGR
    
    Args:
        color: Tupla (B, G, R)
    
    Returns:
        Tupla validada con valores en rango 0-255
    """
    return tuple(max(0, min(255, c)) for c in color)


def validate_contour(contour: np.ndarray, min_points: int = 3) -> bool:
    """
    Valida que un contorno sea válido
    
    Args:
        contour: Contorno a validar
        min_points: Número mínimo de puntos
    
    Returns:
        True si es válido
    """
    if contour is None or len(contour) < min_points:
        return False
    return True
