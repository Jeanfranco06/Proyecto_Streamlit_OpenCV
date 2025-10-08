"""
Widgets interactivos personalizados para los ejercicios
"""
import streamlit as st
import numpy as np
from typing import Optional, Tuple, List


def control_slider(
    etiqueta: str,
    min_valor: int,
    max_valor: int,
    valor_default: int,
    ayuda: str = None,
    key: str = None
) -> int:
    """
    Crea un slider con estilo consistente.
    
    Args:
        etiqueta: Etiqueta del slider
        min_valor: Valor mínimo
        max_valor: Valor máximo
        valor_default: Valor por defecto
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Valor seleccionado
    """
    return st.slider(
        etiqueta,
        min_value=min_valor,
        max_value=max_valor,
        value=valor_default,
        help=ayuda,
        key=key
    )


def control_slider_rango(
    etiqueta: str,
    min_valor: int,
    max_valor: int,
    valor_default: Tuple[int, int],
    ayuda: str = None,
    key: str = None
) -> Tuple[int, int]:
    """
    Crea un slider de rango.
    
    Args:
        etiqueta: Etiqueta del slider
        min_valor: Valor mínimo
        max_valor: Valor máximo
        valor_default: Tupla con valores por defecto (min, max)
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Tupla con valores seleccionados (min, max)
    """
    return st.slider(
        etiqueta,
        min_value=min_valor,
        max_value=max_valor,
        value=valor_default,
        help=ayuda,
        key=key
    )


def selector_color(
    etiqueta: str,
    color_default: str = "#FF0000",
    key: str = None
) -> str:
    """
    Crea un selector de color.
    
    Args:
        etiqueta: Etiqueta del selector
        color_default: Color por defecto en formato hex
        key: Key única para el widget
    
    Returns:
        Color seleccionado en formato hex
    """
    return st.color_picker(etiqueta, value=color_default, key=key)


def selector_opciones(
    etiqueta: str,
    opciones: List[str],
    index_default: int = 0,
    ayuda: str = None,
    key: str = None
) -> str:
    """
    Crea un selectbox con opciones.
    
    Args:
        etiqueta: Etiqueta del selector
        opciones: Lista de opciones
        index_default: Índice de la opción por defecto
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Opción seleccionada
    """
    return st.selectbox(
        etiqueta,
        options=opciones,
        index=index_default,
        help=ayuda,
        key=key
    )


def selector_radio(
    etiqueta: str,
    opciones: List[str],
    index_default: int = 0,
    horizontal: bool = False,
    key: str = None
) -> str:
    """
    Crea un grupo de radio buttons.
    
    Args:
        etiqueta: Etiqueta del selector
        opciones: Lista de opciones
        index_default: Índice de la opción por defecto
        horizontal: Si mostrar las opciones horizontalmente
        key: Key única para el widget
    
    Returns:
        Opción seleccionada
    """
    return st.radio(
        etiqueta,
        options=opciones,
        index=index_default,
        horizontal=horizontal,
        key=key
    )


def checkbox_simple(
    etiqueta: str,
    valor_default: bool = False,
    ayuda: str = None,
    key: str = None
) -> bool:
    """
    Crea un checkbox simple.
    
    Args:
        etiqueta: Etiqueta del checkbox
        valor_default: Valor por defecto
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Estado del checkbox
    """
    return st.checkbox(etiqueta, value=valor_default, help=ayuda, key=key)


def entrada_numero(
    etiqueta: str,
    min_valor: float,
    max_valor: float,
    valor_default: float,
    paso: float = 1.0,
    formato: str = None,
    ayuda: str = None,
    key: str = None
) -> float:
    """
    Crea un input numérico.
    
    Args:
        etiqueta: Etiqueta del input
        min_valor: Valor mínimo
        max_valor: Valor máximo
        valor_default: Valor por defecto
        paso: Incremento del valor
        formato: Formato de visualización
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Valor ingresado
    """
    # Si hay texto de ayuda, incluirlo en la etiqueta
    if ayuda:
        etiqueta = f"{etiqueta} ℹ️\n{ayuda}"
    
    return st.number_input(
        etiqueta,
        min_value=min_valor,
        max_value=max_valor,
        value=valor_default,
        step=paso,
        format=formato,
        key=key
    )


def subir_archivo(
    tipos_aceptados: List[str],
    etiqueta: str = "Sube una imagen",
    ayuda: str = None,
    key: str = None
):
    """
    Crea un widget para subir archivos.
    
    Args:
        tipos_aceptados: Lista de extensiones aceptadas (ej: ['png', 'jpg'])
        etiqueta: Etiqueta del widget
        ayuda: Texto de ayuda opcional
        key: Key única para el widget
    
    Returns:
        Archivo subido o None
    """
    return st.file_uploader(
        etiqueta,
        type=tipos_aceptados,
        help=ayuda,
        key=key
    )


def boton_accion(
    etiqueta: str,
    tipo: str = "primary",
    ancho_completo: bool = False,
    key: str = None
) -> bool:
    """
    Crea un botón de acción.
    
    Args:
        etiqueta: Texto del botón
        tipo: Tipo de botón ('primary' o 'secondary')
        ancho_completo: Si usar el ancho completo del contenedor
        key: Key única para el widget
    
    Returns:
        True si fue presionado, False en caso contrario
    """
    return st.button(
        etiqueta,
        type=tipo,
        use_container_width=ancho_completo,
        key=key
    )


def panel_control(titulo: str = "Controles"):
    """
    Crea un contexto para agrupar controles en un expander.
    
    Args:
        titulo: Título del panel
    
    Returns:
        Contexto del expander
    """
    return st.expander(f"⚙️ {titulo}", expanded=True)


def mostrar_parametros(parametros: dict):
    """
    Muestra un resumen de parámetros seleccionados.
    
    Args:
        parametros: Diccionario con los parámetros a mostrar
    """
    with st.expander("📊 Parámetros Actuales", expanded=False):
        for key, value in parametros.items():
            st.text(f"{key}: {value}")


def boton_descarga(
    datos,
    nombre_archivo: str,
    etiqueta: str = "Descargar resultado",
    tipo_mime: str = "image/png",
    key: str = None
):
    """
    Crea un botón para descargar datos.
    
    Args:
        datos: Datos a descargar (bytes)
        nombre_archivo: Nombre del archivo a descargar
        etiqueta: Texto del botón
        tipo_mime: Tipo MIME del archivo
        key: Key única para el widget
    """
    st.download_button(
        label=etiqueta,
        data=datos,
        file_name=nombre_archivo,
        mime=tipo_mime,
        key=key
    )


def selector_coordenadas() -> Optional[Tuple[int, int, int, int]]:
    """
    Crea controles para seleccionar coordenadas de un rectángulo.
    
    Returns:
        Tupla (x1, y1, x2, y2) o None
    """
    with panel_control("Seleccionar Región"):
        col1, col2 = st.columns(2)
        
        with col1:
            x1 = st.number_input("X inicial", min_value=0, value=0, step=1)
            y1 = st.number_input("Y inicial", min_value=0, value=0, step=1)
        
        with col2:
            x2 = st.number_input("X final", min_value=0, value=100, step=1)
            y2 = st.number_input("Y final", min_value=0, value=100, step=1)
        
        if x2 <= x1 or y2 <= y1:
            st.warning("Las coordenadas finales deben ser mayores que las iniciales")
            return None
        
        return (int(x1), int(y1), int(x2), int(y2))


def info_tooltip(texto: str):
    """
    Muestra un tooltip informativo.
    
    Args:
        texto: Texto a mostrar
    """
    st.info(f"💡 {texto}")


def warning_tooltip(texto: str):
    """
    Muestra un tooltip de advertencia.
    
    Args:
        texto: Texto a mostrar
    """
    st.warning(f"⚠️ {texto}")