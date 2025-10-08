"""
Funciones de utilidad global para el proyecto OpenCV + Streamlit
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import streamlit as st
from PIL import Image


def leer_imagen(ruta: Union[str, Path], modo: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Lee una imagen desde un archivo usando OpenCV.
    
    Args:
        ruta: Ruta al archivo de imagen
        modo: Modo de lectura (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, etc.)
    
    Returns:
        Imagen como array numpy o None si falla
    """
    try:
        ruta = str(ruta)
        img = cv2.imread(ruta, modo)
        if img is None:
            st.error(f"No se pudo leer la imagen: {ruta}")
            return None
        return img
    except Exception as e:
        st.error(f"Error al leer imagen: {str(e)}")
        return None


def guardar_imagen(img: np.ndarray, ruta: Union[str, Path]) -> bool:
    """
    Guarda una imagen usando OpenCV.
    
    Args:
        img: Imagen como array numpy
        ruta: Ruta donde guardar la imagen
    
    Returns:
        True si se guardó correctamente, False en caso contrario
    """
    try:
        ruta = str(ruta)
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(ruta, img)
    except Exception as e:
        st.error(f"Error al guardar imagen: {str(e)}")
        return False


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen de BGR (OpenCV) a RGB (Streamlit/Matplotlib).
    
    Args:
        img: Imagen en formato BGR
    
    Returns:
        Imagen en formato RGB
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen de RGB a BGR.
    
    Args:
        img: Imagen en formato RGB
    
    Returns:
        Imagen en formato BGR
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def redimensionar_imagen(
    img: np.ndarray,
    ancho: Optional[int] = None,
    alto: Optional[int] = None,
    mantener_aspecto: bool = True
) -> np.ndarray:
    """
    Redimensiona una imagen.
    
    Args:
        img: Imagen a redimensionar
        ancho: Ancho deseado (None para calcularlo automáticamente)
        alto: Alto deseado (None para calcularlo automáticamente)
        mantener_aspecto: Si mantener la relación de aspecto
    
    Returns:
        Imagen redimensionada
    """
    h, w = img.shape[:2]
    
    if ancho is None and alto is None:
        return img
    
    if mantener_aspecto:
        if ancho is None:
            ratio = alto / h
            ancho = int(w * ratio)
        elif alto is None:
            ratio = ancho / w
            alto = int(h * ratio)
    else:
        if ancho is None:
            ancho = w
        if alto is None:
            alto = h
    
    return cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA)


def mostrar_imagen_streamlit(
    img: np.ndarray,
    caption: str = "",
    use_column_width: bool = True,
    convertir_rgb: bool = True
):
    """
    Muestra una imagen en Streamlit, manejando la conversión BGR->RGB automáticamente.
    
    Args:
        img: Imagen a mostrar
        caption: Título de la imagen
        use_column_width: Si ajustar al ancho de la columna
        convertir_rgb: Si convertir de BGR a RGB (True para imágenes de OpenCV)
    """
    if img is None:
        st.warning("No hay imagen para mostrar")
        return
    
    img_mostrar = bgr_to_rgb(img) if convertir_rgb and len(img.shape) == 3 else img
    st.image(img_mostrar, caption=caption, use_column_width=use_column_width)


def comparar_imagenes(
    img_original: np.ndarray,
    img_procesada: np.ndarray,
    titulos: Tuple[str, str] = ("Original", "Procesada")
):
    """
    Muestra dos imágenes lado a lado para comparación.
    
    Args:
        img_original: Imagen original
        img_procesada: Imagen procesada
        titulos: Tupla con los títulos de cada imagen
    """
    col1, col2 = st.columns(2)
    
    with col1:
        mostrar_imagen_streamlit(img_original, caption=titulos[0])
    
    with col2:
        mostrar_imagen_streamlit(img_procesada, caption=titulos[1])


def cargar_imagen_desde_upload(archivo_subido) -> Optional[np.ndarray]:
    """
    Carga una imagen desde un archivo subido por el usuario en Streamlit.
    
    Args:
        archivo_subido: Objeto UploadedFile de Streamlit
    
    Returns:
        Imagen como array numpy en formato BGR (OpenCV) o None si falla
    """
    if archivo_subido is None:
        return None
    
    try:
        # Leer el archivo como imagen PIL
        imagen_pil = Image.open(archivo_subido)
        
        # Convertir a array numpy
        imagen_np = np.array(imagen_pil)
        
        # Convertir a BGR si es RGB (para compatibilidad con OpenCV)
        if len(imagen_np.shape) == 3 and imagen_np.shape[2] == 3:
            imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
        
        return imagen_np
    except Exception as e:
        st.error(f"Error al cargar imagen: {str(e)}")
        return None


def obtener_dimensiones(img: np.ndarray) -> Tuple[int, int, int]:
    """
    Obtiene las dimensiones de una imagen.
    
    Args:
        img: Imagen como array numpy
    
    Returns:
        Tupla (alto, ancho, canales). Si es escala de grises, canales = 1
    """
    if len(img.shape) == 2:
        h, w = img.shape
        return h, w, 1
    else:
        h, w, c = img.shape
        return h, w, c


def crear_grid_imagenes(imagenes: list, titulos: list = None, cols: int = 3):
    """
    Muestra múltiples imágenes en una cuadrícula.
    
    Args:
        imagenes: Lista de imágenes a mostrar
        titulos: Lista de títulos (opcional)
        cols: Número de columnas en la cuadrícula
    """
    if titulos is None:
        titulos = [f"Imagen {i+1}" for i in range(len(imagenes))]
    
    # Crear filas de columnas
    for i in range(0, len(imagenes), cols):
        columnas = st.columns(cols)
        for j, col in enumerate(columnas):
            idx = i + j
            if idx < len(imagenes):
                with col:
                    mostrar_imagen_streamlit(imagenes[idx], caption=titulos[idx])


def mostrar_info_imagen(img: np.ndarray):
    """
    Muestra información técnica de una imagen en un expander.
    
    Args:
        img: Imagen a analizar
    """
    h, w, c = obtener_dimensiones(img)
    
    with st.expander("ℹ️ Información de la imagen"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Alto", f"{h} px")
        with col2:
            st.metric("Ancho", f"{w} px")
        with col3:
            st.metric("Canales", c)
        
        st.write(f"**Tipo de datos:** {img.dtype}")
        st.write(f"**Forma:** {img.shape}")
        st.write(f"**Tamaño total:** {img.size:,} elementos")