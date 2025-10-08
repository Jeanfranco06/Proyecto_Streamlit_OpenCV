"""
Componentes de layout principal (header, footer, estructura)
"""
import streamlit as st
from core.config import APP_TITLE, APP_ICON, APP_VERSION


def renderizar_header():
    """Renderiza el encabezado principal de la aplicación."""
    st.markdown(f"""
    <div style='text-align: center; padding: 1.5rem 0 2rem 0;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>
            {APP_ICON} {APP_TITLE}
        </h1>
        <p style='color: #666; margin-top: 0.5rem; font-size: 0.9rem;'>
            Tutorial Interactivo de Visión por Computadora
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")


def renderizar_footer():
    """Renderiza el pie de página de la aplicación."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0; color: #666; font-size: 0.85rem;'>
            <p style='margin: 0;'>
                Tutorial de OpenCV • Versión {APP_VERSION}
            </p>
            <p style='margin: 0.5rem 0 0 0;'>
                Hecho con Streamlit y OpenCV
            </p>
        </div>
        """, unsafe_allow_html=True)


def crear_seccion(titulo: str, icono: str = ""):
    """
    Crea una sección con título estilizado.
    
    Args:
        titulo: Título de la sección
        icono: Icono a mostrar junto al título
    """
    st.markdown(f"""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <h3 style='margin: 0; color: #1f77b4;'>
            {icono} {titulo}
        </h3>
    </div>
    """, unsafe_allow_html=True)


def crear_alerta(mensaje: str, tipo: str = "info"):
    """
    Crea una alerta personalizada.
    
    Args:
        mensaje: Mensaje a mostrar
        tipo: Tipo de alerta ('info', 'success', 'warning', 'error')
    """
    colores = {
        "info": "#d1ecf1",
        "success": "#d4edda",
        "warning": "#fff3cd",
        "error": "#f8d7da"
    }
    
    iconos = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    color = colores.get(tipo, colores["info"])
    icono = iconos.get(tipo, iconos["info"])
    
    st.markdown(f"""
    <div style='background-color: {color}; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <p style='margin: 0;'>
            {icono} {mensaje}
        </p>
    </div>
    """, unsafe_allow_html=True)


def crear_card(titulo: str, contenido: str, icono: str = "📄"):
    """
    Crea una tarjeta (card) con título y contenido.
    
    Args:
        titulo: Título de la tarjeta
        contenido: Contenido de la tarjeta
        icono: Icono a mostrar
    """
    st.markdown(f"""
    <div style='border: 1px solid #ddd; border-radius: 0.5rem; padding: 1.5rem; margin: 1rem 0;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #1f77b4;'>
            {icono} {titulo}
        </h4>
        <p style='margin: 0; color: #555;'>
            {contenido}
        </p>
    </div>
    """, unsafe_allow_html=True)


def mostrar_codigo(codigo: str, lenguaje: str = "python"):
    """
    Muestra código con sintaxis resaltada en un expander.
    
    Args:
        codigo: Código a mostrar
        lenguaje: Lenguaje de programación
    """
    with st.expander("👨‍💻 Ver código", expanded=False):
        st.code(codigo, language=lenguaje)


def crear_tabs(titulos: list):
    """
    Crea tabs personalizadas.
    
    Args:
        titulos: Lista de títulos para las tabs
    
    Returns:
        Lista de objetos tab de Streamlit
    """
    return st.tabs(titulos)


def mostrar_barra_progreso(valor: int, maximo: int, texto: str = ""):
    """
    Muestra una barra de progreso.
    
    Args:
        valor: Valor actual
        maximo: Valor máximo
        texto: Texto opcional a mostrar
    """
    progreso = valor / maximo if maximo > 0 else 0
    
    if texto:
        st.caption(texto)
    
    st.progress(progreso)


def crear_metricas(metricas: dict):
    """
    Crea una fila de métricas.
    
    Args:
        metricas: Diccionario con formato {label: value}
    """
    cols = st.columns(len(metricas))
    
    for col, (label, valor) in zip(cols, metricas.items()):
        with col:
            st.metric(label, valor)


def separador(texto: str = ""):
    """
    Crea un separador visual con texto opcional.
    
    Args:
        texto: Texto opcional a mostrar en el separador
    """
    if texto:
        st.markdown(f"**{texto}**")
    st.markdown("---")