"""
Barra lateral de navegación para el tutorial de OpenCV
"""
import streamlit as st
from typing import Tuple, Optional
import re


def renderizar_sidebar(router) -> Tuple[Optional[str], Optional[str]]:
    """
    Renderiza la barra lateral con selección de capítulos y ejercicios.
    """
    with st.sidebar:
        st.title("📚 Navegación")
        st.markdown("---")
        
        # Verificar si hay ejercicios disponibles
        if not router.tiene_ejercicios():
            st.warning("⚠️ No se encontraron ejercicios")
            st.info("Asegúrate de tener ejercicios en la carpeta `exercises/`")
            return None, None
        
        # Obtener estadísticas
        stats = router.obtener_estadisticas()
        
        with st.expander("ℹ️ Información", expanded=False):
            st.metric("Total Capítulos", stats['total_capitulos'])
            st.metric("Total Ejercicios", stats['total_ejercicios'])
        
        st.markdown("---")
        
        # Obtener capítulos
        capitulos = router.obtener_capitulos()
        if not capitulos:
            st.error("No hay capítulos disponibles")
            return None, None
        
        # Formatear nombres
        capitulos_formateados = [_formatear_nombre_capitulo(cap) for cap in capitulos]

        # ✅ Ordenar los capítulos y sus nombres formateados juntos
        capitulos_ordenados_y_formateados = sorted(
            zip(capitulos, capitulos_formateados),
            key=lambda x: numero_capitulo(x[0])
        )
        capitulos_ordenados, capitulos_formateados_ordenados = zip(*capitulos_ordenados_y_formateados)

        # Inicializar estado
        if 'capitulo_idx' not in st.session_state:
            st.session_state.capitulo_idx = 0
        if 'ejercicio_idx' not in st.session_state:
            st.session_state.ejercicio_idx = 0
        
        # ✅ Mostrar selectbox con capítulos ordenados
        capitulo_idx = st.selectbox(
            "🗂️ Selecciona un Capítulo",
            range(len(capitulos_ordenados)),
            format_func=lambda i: capitulos_formateados_ordenados[i],
            key='capitulo_select',
            index=st.session_state.capitulo_idx
        )

        # ✅ Actualizar session_state correctamente
        if capitulo_idx != st.session_state.capitulo_idx:
            st.session_state.capitulo_idx = capitulo_idx
            st.session_state.ejercicio_idx = 0
        
        # ✅ Ahora sí usamos la lista ordenada
        capitulo_seleccionado = capitulos_ordenados[capitulo_idx]
        
        # Obtener ejercicios
        ejercicios = router.obtener_ejercicios(capitulo_seleccionado)
        if not ejercicios:
            st.warning(f"No hay ejercicios en {capitulos_formateados_ordenados[capitulo_idx]}")
            return capitulo_seleccionado, None
        
        st.markdown("---")
        st.caption(f"📝 {len(ejercicios)} ejercicio(s) disponible(s)")
        
        # Selectbox para ejercicio
        ejercicio_idx = st.selectbox(
            "📋 Selecciona un Ejercicio",
            range(len(ejercicios)),
            format_func=lambda i: ejercicios[i]['titulo'],
            key='ejercicio_select',
            index=st.session_state.ejercicio_idx
        )
        
        st.session_state.ejercicio_idx = ejercicio_idx
        ejercicio_seleccionado = ejercicios[ejercicio_idx]['nombre']
        
        st.markdown("---")
        
        # Navegación
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Anterior", use_container_width=True):
                _navegar_anterior(ejercicios, capitulos_ordenados)
                st.rerun()
        with col2:
            if st.button("Siguiente ➡️", use_container_width=True):
                _navegar_siguiente(ejercicios, capitulos_ordenados)
                st.rerun()
        
        st.markdown("---")
        _mostrar_info_adicional()
        
        return capitulo_seleccionado, ejercicio_seleccionado


# ======================================
# 🔧 FUNCIONES AUXILIARES
# ======================================

def numero_capitulo(nombre):
    match = re.search(r'\d+', nombre)
    return int(match.group()) if match else 0


def _formatear_nombre_capitulo(nombre: str) -> str:
    if nombre.startswith("chapter"):
        numero = nombre.replace("chapter", "")
        return f"Capítulo {numero}"
    elif nombre.startswith("cap"):
        numero = nombre.replace("cap", "")
        return f"Capítulo {numero}"
    elif nombre.startswith("tema"):
        numero = nombre.replace("tema_", "").replace("tema", "")
        return f"Tema {numero}"
    else:
        return nombre.replace("_", " ").title()


def _navegar_anterior(ejercicios, capitulos):
    if st.session_state.ejercicio_idx > 0:
        st.session_state.ejercicio_idx -= 1
    elif st.session_state.capitulo_idx > 0:
        st.session_state.capitulo_idx -= 1
        st.session_state.ejercicio_idx = 0


def _navegar_siguiente(ejercicios, capitulos):
    if st.session_state.ejercicio_idx < len(ejercicios) - 1:
        st.session_state.ejercicio_idx += 1
    elif st.session_state.capitulo_idx < len(capitulos) - 1:
        st.session_state.capitulo_idx += 1
        st.session_state.ejercicio_idx = 0


def _mostrar_info_adicional():
    with st.expander("💡 Consejos", expanded=False):
        st.markdown("""
        - Usa los controles interactivos  
        - Experimenta con los parámetros  
        - Lee el código de ejemplo  
        - Observa los resultados visuales
        """)
    
    with st.expander("Configuración", expanded=False):
        debug_mode = st.checkbox(
            "Modo Debug",
            value=st.session_state.get("debug_mode", False),
            help="Recarga los módulos en cada ejecución"
        )
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.caption("Modo debug activado")
