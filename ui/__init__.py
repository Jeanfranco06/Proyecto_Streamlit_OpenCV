"""
Paquete UI - Componentes de interfaz de usuario para el tutorial de OpenCV
"""

from .sidebar import renderizar_sidebar
from .layout import (
    renderizar_header,
    renderizar_footer,
    crear_seccion,
    crear_alerta,
    crear_card,
    mostrar_codigo,
    crear_tabs,
    mostrar_barra_progreso,
    crear_metricas,
    separador
)
from .widgets import (
    control_slider,
    control_slider_rango,
    selector_color,
    selector_opciones,
    selector_radio,
    checkbox_simple,
    entrada_numero,
    subir_archivo,
    boton_accion,
    panel_control,
    mostrar_parametros,
    boton_descarga,
    selector_coordenadas,
    info_tooltip,
    warning_tooltip
)

__all__ = [
    # Sidebar
    'renderizar_sidebar',
    
    # Layout
    'renderizar_header',
    'renderizar_footer',
    'crear_seccion',
    'crear_alerta',
    'crear_card',
    'mostrar_codigo',
    'crear_tabs',
    'mostrar_barra_progreso',
    'crear_metricas',
    'separador',
    
    # Widgets
    'control_slider',
    'control_slider_rango',
    'selector_color',
    'selector_opciones',
    'selector_radio',
    'checkbox_simple',
    'entrada_numero',
    'subir_archivo',
    'boton_accion',
    'panel_control',
    'mostrar_parametros',
    'boton_descarga',
    'selector_coordenadas',
    'info_tooltip',
    'warning_tooltip',
]