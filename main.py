
import streamlit as st
from core.config import APP_TITLE, APP_ICON, PAGE_CONFIG
from core.router import RouterEjercicio
from ui.sidebar import renderizar_sidebar
from ui.layout import renderizar_header, renderizar_footer

st.set_page_config(**PAGE_CONFIG)

def main():
    
    router = RouterEjercicio()
    
    renderizar_header()
    
    capitulo_seleccionado, ejercicio_seleccionado = renderizar_sidebar(router)
    
    main_container = st.container()
    
    with main_container:
        if capitulo_seleccionado and ejercicio_seleccionado:
            modulo_de_ejercicio = router.cargar_ejercicio(capitulo_seleccionado, ejercicio_seleccionado)
            
            if modulo_de_ejercicio:
                try:
                    modulo_de_ejercicio.run()
                except Exception as e:
                    st.error(f"Error al ejecutar el ejercicio! - {str(e)}")
                    with st.expander("Detalles del error"):
                        st.code(str(e))
            else:
                st.warning("No se pudo cargar el ejercicio seleccionado")
        else:
            st.markdown("""
            <div style='text-align: center; padding: 3rem 0;'>
                <h2>👋 Bienvenido al Tutorial Interactivo de OpenCV</h2>
                <p style='font-size: 1.1rem; color: #666; margin-top: 1rem;'>
                    Selecciona un capítulo y ejercicio del menú lateral para comenzar
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Aprendizaje Interactivo**\n\nEjemplos prácticos con código ejecutable")
            
            with col2:
                st.success("**Ejercicios Guiados**\n\nDesde básico hasta avanzado")
            
            with col3:
                st.warning("**Resultados en Tiempo Real**\n\nVisualiza los resultados al instante")
    
    renderizar_footer()

if __name__ == "__main__":
    main()