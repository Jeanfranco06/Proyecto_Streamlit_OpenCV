"""
Capítulo 3 - Ejercicio 3: Cartoonizar una Imagen
Aprende a transformar fotografías en dibujos animados usando filtros y detección de bordes
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from core.utils import (
    leer_imagen,
    bgr_to_rgb,
    mostrar_imagen_streamlit,
    comparar_imagenes,
    cargar_imagen_desde_upload
)
from ui.layout import crear_seccion, mostrar_codigo, crear_alerta, crear_tabs
from ui.widgets import (
    control_slider,
    panel_control,
    checkbox_simple,
    selector_opciones,
    boton_accion,
    info_tooltip,
    entrada_numero
)


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Cartoonizar una Imagen")
    st.markdown("""
    Transforma fotografías realistas en dibujos animados estilizados usando técnicas de 
    procesamiento de imágenes: detección de bordes, filtros bilaterales y operaciones de máscaras.
    """)
    
    st.markdown("---")
    
    # Cargar imagen
    img = cargar_imagen_input()
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cartoonizar Interactivo",
        "Estilos Predefinidos",
        "Proceso Paso a Paso",
        "Teoría"
    ])
    
    with tab1:
        modo_interactivo(img)
    
    with tab2:
        estilos_predefinidos(img)
    
    with tab3:
        proceso_paso_a_paso(img)
    
    with tab4:
        mostrar_teoria()


def modo_interactivo(img):
    """Modo interactivo con controles ajustables en tiempo real."""
    
    crear_seccion("Controles de Cartoonización", "")
    
    col_control, col_preview = st.columns([1, 2])
    
    with col_control:
        with panel_control("Parámetros del Efecto"):
            
            # Modo de renderizado
            modo_render = selector_opciones(
                "Modo de Renderizado",
                ["Cartoon a Color", "Sketch (Solo Bordes)", "Comparación Lado a Lado"],
                key="modo_render"
            )
            
            st.markdown("---")
            st.markdown("**Detección de Bordes**")
            
            # Tamaño del kernel Laplacian
            ksize = selector_opciones(
                "Tamaño de Kernel",
                ["1", "3", "5", "7"],
                index_default=2,
                ayuda="Kernel más grande = bordes más gruesos",
                key="ksize"
            )
            ksize = int(ksize)
            
            # Umbral de detección de bordes
            threshold_value = control_slider(
                "Umbral de Bordes",
                50, 200, 100,
                "Controla la sensibilidad de detección de bordes",
                key="threshold"
            )
            
            # Tamaño del median blur
            median_blur = selector_opciones(
                "Median Blur",
                ["3", "5", "7", "9"],
                index_default=2,
                ayuda="Reduce ruido antes de detectar bordes",
                key="median_blur"
            )
            median_blur = int(median_blur)
            
            st.markdown("---")
            st.markdown("**Filtro Bilateral**")
            
            # Número de repeticiones del filtro bilateral
            num_repetitions = control_slider(
                "Repeticiones de Filtro",
                1, 15, 10,
                "Más repeticiones = efecto más cartoon",
                key="repetitions"
            )
            
            # Sigma color
            sigma_color = control_slider(
                "Sigma Color",
                1, 20, 5,
                "Rango de colores a considerar",
                key="sigma_color"
            )
            
            # Sigma space
            sigma_space = control_slider(
                "Sigma Space",
                1, 20, 7,
                "Rango espacial del filtro",
                key="sigma_space"
            )
            
            st.markdown("---")
            st.markdown("**🔧 Optimización**")
            
            # Factor de downsampling
            ds_factor = control_slider(
                "Factor de Reducción",
                1, 8, 4,
                "Reduce tamaño para procesamiento más rápido",
                key="ds_factor"
            )
            
            st.markdown("---")
            
            # Opciones de visualización
            mostrar_mascaras = checkbox_simple(
                "Mostrar máscara de bordes",
                False,
                key="show_mask"
            )
    
    with col_preview:
        crear_seccion("Resultado", "")
        
        # Aplicar cartoonización
        if modo_render == "Sketch (Solo Bordes)":
            img_cartoon = cartoonize_image(
                img, ksize, True, 
                num_repetitions, sigma_color, sigma_space, 
                ds_factor, threshold_value, median_blur
            )
            mostrar_imagen_streamlit(img_cartoon, "Modo Sketch")
            
        elif modo_render == "Cartoon a Color":
            img_cartoon = cartoonize_image(
                img, ksize, False,
                num_repetitions, sigma_color, sigma_space,
                ds_factor, threshold_value, median_blur
            )
            mostrar_imagen_streamlit(img_cartoon, "Cartoon a Color")
            
        else:  # Comparación
            img_sketch = cartoonize_image(
                img, ksize, True,
                num_repetitions, sigma_color, sigma_space,
                ds_factor, threshold_value, median_blur
            )
            img_color = cartoonize_image(
                img, ksize, False,
                num_repetitions, sigma_color, sigma_space,
                ds_factor, threshold_value, median_blur
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original**")
                mostrar_imagen_streamlit(img, "")
            with col2:
                st.markdown("**Sketch**")
                mostrar_imagen_streamlit(img_sketch, "")
            with col3:
                st.markdown("**Cartoon Color**")
                mostrar_imagen_streamlit(img_color, "")
        
        # Mostrar máscara si está activado
        if mostrar_mascaras:
            st.markdown("---")
            crear_seccion("Máscara de Bordes", "")
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.medianBlur(img_gray, median_blur)
            edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
            ret, mask = cv2.threshold(edges, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Bordes Detectados (Laplacian)**")
                mostrar_imagen_streamlit(
                    cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                    "",
                    convertir_rgb=False
                )
            with col2:
                st.markdown("**Máscara Final (Umbralizada)**")
                mostrar_imagen_streamlit(
                    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                    "",
                    convertir_rgb=False
                )
        
        # Botón de descarga
        if modo_render != "Comparación Lado a Lado":
            if boton_accion("Guardar resultado", key="save_interactive"):
                guardar_resultado(img_cartoon, "cartoon_custom.jpg")


def estilos_predefinidos(img):
    """Presets de estilos cartoon populares."""
    
    crear_seccion("Estilos Predefinidos", "")
    
    st.markdown("""
    Selecciona un estilo preconfigurado inspirado en diferentes estilos de animación y cómic.
    """)
    
    # Definir presets
    presets = {
        "Comic Clásico": {
            "ksize": 5, "threshold": 100, "median": 7,
            "reps": 10, "sigma_c": 5, "sigma_s": 7, "ds": 4,
            "modo": False
        },
        "Manga Japonés": {
            "ksize": 3, "threshold": 120, "median": 5,
            "reps": 8, "sigma_c": 8, "sigma_s": 8, "ds": 3,
            "modo": False
        },
        "Sketch Artístico": {
            "ksize": 7, "threshold": 80, "median": 9,
            "reps": 5, "sigma_c": 3, "sigma_s": 5, "ds": 2,
            "modo": True
        },
        "Animación Disney": {
            "ksize": 5, "threshold": 90, "median": 7,
            "reps": 12, "sigma_c": 9, "sigma_s": 9, "ds": 4,
            "modo": False
        },
        "Ilustración Infantil": {
            "ksize": 7, "threshold": 110, "median": 9,
            "reps": 15, "sigma_c": 12, "sigma_s": 10, "ds": 5,
            "modo": False
        },
        "Boceto a Lápiz": {
            "ksize": 3, "threshold": 100, "median": 5,
            "reps": 3, "sigma_c": 2, "sigma_s": 3, "ds": 2,
            "modo": True
        },
    }
    
    # Descripciones
    descripciones = {
        "Comic Clásico": "Estilo equilibrado con bordes definidos y colores planos, perfecto para cómics occidentales",
        "Manga Japonés": "Bordes finos y detalles sutiles característicos del manga",
        "Sketch Artístico": "Solo líneas, perfecto para bocetos artísticos en blanco y negro",
        "Animación Disney": "Colores vibrantes y suaves, estilo de animación tradicional",
        "Ilustración Infantil": "Muy suavizado con colores brillantes, ideal para libros infantiles",
        "Boceto a Lápiz": "Líneas delicadas simulando un dibujo a lápiz",
    }
    
    # Selector de preset
    preset_seleccionado = selector_opciones(
        "Selecciona un estilo",
        list(presets.keys()),
        key="preset_style_cartoon"
    )
    
    params = presets[preset_seleccionado]
    
    # Mostrar descripción
    info_tooltip(descripciones[preset_seleccionado])
    
    st.markdown("---")
    
    # Aplicar preset
    img_cartoon = cartoonize_image(
        img,
        params["ksize"],
        params["modo"],
        params["reps"],
        params["sigma_c"],
        params["sigma_s"],
        params["ds"],
        params["threshold"],
        params["median"]
    )
    
    # Mostrar resultado
    crear_seccion("Vista Previa", "")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown(f"**✨ {preset_seleccionado}**")
        mostrar_imagen_streamlit(img_cartoon, "")
    
    # Botón de descarga
    if boton_accion("Guardar estilo", key="save_preset_cartoon"):
        guardar_resultado(img_cartoon, f"cartoon_{preset_seleccionado.lower().replace(' ', '_')}.jpg")


def proceso_paso_a_paso(img):
    """Visualización paso a paso del proceso de cartoonización."""
    
    crear_seccion("Proceso Paso a Paso", "")
    
    st.markdown("""
    Observa cada etapa del proceso de transformación de una fotografía a cartoon.
    """)
    
    # Parámetros para el ejemplo
    ksize = 5
    threshold_val = 100
    median_blur = 7
    num_reps = 10
    sigma_c = 5
    sigma_s = 7
    ds_factor = 4
    
    # Paso 1: Conversión a escala de grises
    st.markdown("###  Paso 1: Conversión a Escala de Grises")
    st.markdown("Convertimos la imagen a escala de grises para facilitar la detección de bordes.")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original (Color)**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown("**Escala de Grises**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    
    st.markdown("---")
    
    # Paso 2: Median Blur
    st.markdown("### Paso 2: Aplicar Median Blur")
    st.markdown("Reducimos el ruido para obtener bordes más limpios.")
    
    img_blur = cv2.medianBlur(img_gray, median_blur)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Antes del Blur**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col2:
        st.markdown(f"**Después Median Blur ({median_blur}x{median_blur})**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    
    st.markdown("---")
    
    # Paso 3: Detección de bordes con Laplacian
    st.markdown("### Paso 3: Detección de Bordes (Laplacian)")
    st.markdown("Detectamos los bordes usando el operador Laplaciano.")
    
    edges = cv2.Laplacian(img_blur, cv2.CV_8U, ksize=ksize)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen Suavizada**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col2:
        st.markdown(f"**Bordes Detectados (Kernel {ksize}x{ksize})**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    
    st.markdown("---")
    
    # Paso 4: Umbralización
    st.markdown("### Paso 4: Umbralización (Thresholding)")
    st.markdown("Convertimos los bordes a una máscara binaria.")
    
    ret, mask = cv2.threshold(edges, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bordes (Valores Continuos)**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col2:
        st.markdown(f"**Máscara Binaria (Umbral={threshold_val})**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    
    st.info(f"Píxeles blancos (bordes): {np.count_nonzero(mask == 255):,} | "
            f"Píxeles negros: {np.count_nonzero(mask == 0):,}")
    
    st.markdown("---")
    
    # Paso 5: Downsampling
    st.markdown("### Paso 5: Reducción de Tamaño (Downsampling)")
    st.markdown(f"Reducimos la imagen {ds_factor}x para procesar más rápido.")
    
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, 
                          interpolation=cv2.INTER_AREA)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Original ({img.shape[1]}x{img.shape[0]})**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown(f"**Reducida ({img_small.shape[1]}x{img_small.shape[0]})**")
        mostrar_imagen_streamlit(img_small, "")
    
    st.markdown("---")
    
    # Paso 6: Filtro Bilateral (animación de progreso)
    st.markdown("### Paso 6: Aplicar Filtro Bilateral")
    st.markdown(f"Aplicamos el filtro bilateral {num_reps} veces para suavizar colores.")
    
    # Mostrar progreso con algunas iteraciones
    iteraciones_mostrar = [0, num_reps//2, num_reps]
    
    cols = st.columns(len(iteraciones_mostrar))
    
    img_bilateral = img_small.copy()
    for idx, num_iter in enumerate(iteraciones_mostrar):
        # Aplicar filtro hasta esta iteración
        img_temp = img_small.copy()
        for i in range(num_iter):
            img_temp = cv2.bilateralFilter(img_temp, ksize, sigma_c, sigma_s)
        
        with cols[idx]:
            st.markdown(f"**Iteración {num_iter}**")
            mostrar_imagen_streamlit(img_temp, "")
    
    # Aplicar todas las iteraciones para el resultado final
    for i in range(num_reps):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_c, sigma_s)
    
    st.markdown("---")
    
    # Paso 7: Upsampling
    st.markdown("### Paso 7: Restaurar Tamaño Original (Upsampling)")
    st.markdown(f"Escalamos de vuelta al tamaño original.")
    
    img_output = cv2.resize(img_small, (img.shape[1], img.shape[0]), 
                           interpolation=cv2.INTER_LINEAR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Imagen Pequeña ({img_small.shape[1]}x{img_small.shape[0]})**")
        mostrar_imagen_streamlit(img_small, "")
    with col2:
        st.markdown(f"**Tamaño Original ({img_output.shape[1]}x{img_output.shape[0]})**")
        mostrar_imagen_streamlit(img_output, "")
    
    st.markdown("---")
    
    # Paso 8: Combinar con máscara
    st.markdown("### Paso 8: Aplicar Máscara de Bordes")
    st.markdown("Combinamos la imagen suavizada con los bordes usando operación AND.")
    
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Imagen Suavizada**")
        mostrar_imagen_streamlit(img_output, "")
    with col2:
        st.markdown("**Máscara de Bordes**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col3:
        st.markdown("**Resultado Final (AND)**")
        mostrar_imagen_streamlit(dst, "")
    
    st.markdown("---")
    
    # Comparación final
    crear_seccion("Resultado Final", "")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fotografía Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown("**Imagen Cartoonizada**")
        mostrar_imagen_streamlit(dst, "")


def mostrar_teoria():
    """Explicación teórica del proceso de cartoonización."""
    
    crear_seccion("Teoría: Cartoonización de Imágenes", "")
    
    st.markdown("""
    ### ¿Qué es la Cartoonización?
    
    La **cartoonización** es el proceso de transformar una fotografía realista en una imagen 
    estilizada que parece dibujada a mano, similar a un cómic o dibujo animado. El resultado 
    tiene colores planos, bordes definidos y menos detalles.
    
    ###Características de una Imagen Cartoon
    
    - **Colores planos** - Áreas de color uniforme sin gradientes complejos
    - **Bordes marcados** - Líneas negras gruesas definiendo formas
    - **Menos detalles** - Texturas y variaciones sutiles eliminadas
    - **Contraste alto** - Diferenciación clara entre regiones
    
    ###Pipeline del Algoritmo
    
    El proceso consta de **8 pasos principales**:
    
    ```
    1. Fotografía Original
           ↓
    2. Conversión a Escala de Grises
           ↓
    3. Median Blur (Reducción de Ruido)
           ↓
    4. Detección de Bordes (Laplacian)
           ↓
    5. Umbralización (Máscara Binaria)
           ↓
    6. Downsampling (Optimización)
           ↓
    7. Filtro Bilateral (Múltiples Repeticiones)
           ↓
    8. Upsampling + Aplicación de Máscara
           ↓
    9. Imagen Cartoon Final
    ```
    
    ### Técnicas Utilizadas
    
    #### 1. **Operador Laplaciano**
    
    Detecta bordes calculando la segunda derivada de la imagen:
    
    ```
    ∇²f = ∂²f/∂x² + ∂²f/∂y²
    ```
    
    El kernel Laplaciano típico es:
    ```
    | 0  1  0 |
    | 1 -4  1 |
    | 0  1  0 |
    ```
    
    - **Ventaja**: Detecta bordes en todas las direcciones simultáneamente
    - **Desventaja**: Sensible al ruido (por eso aplicamos median blur primero)
    
    #### 2. **Filtro Bilateral**
    
    Suaviza la imagen preservando bordes. Combina dos kernels Gaussianos:
    
    - **Kernel espacial** (σ_space): Similar al Gaussian blur
    - **Kernel de rango** (σ_color): Considera diferencia de color
    
    ```python
    BF[I]_p = (1/W_p) * Σ(I_q * G_σs(||p-q||) * G_σr(|I_p - I_q|))
    ```
    
    **¿Por qué múltiples repeticiones?**
    - Cada iteración suaviza más los colores
    - Crea áreas de color más uniformes
    - Resultado más "cartoon-like"
    
    #### 3. **Umbralización (Thresholding)**
    
    Convierte bordes continuos en máscara binaria:
    
    ```
    mask(x,y) = { 255  si edge(x,y) < threshold
                { 0    si edge(x,y) ≥ threshold
    ```
    
    Se usa `THRESH_BINARY_INV` para invertir: queremos bordes en **negro** (0).
    
    #### 4. **Operación AND Bitwise**
    
    Combina imagen suavizada con máscara de bordes:
    
    ```
    resultado = imagen_suavizada AND máscara_bordes
    ```
    
    Esto "dibuja" los bordes negros sobre la imagen de colores planos.
    
    ### Dos Modos de Renderizado
    
    | Modo | Descripción | Aplicación |
    |------|-------------|------------|
    | **Sketch** | Solo máscara de bordes en blanco | Bocetos, dibujos a línea |
    | **Cartoon Color** | Imagen suavizada + bordes | Cómics, animación |
    
    ### Parámetros Clave
    
    **Kernel Size (ksize)**
    - 1-3: Bordes muy finos (estilo manga)
    - 5: Equilibrado (estilo cómic clásico)
    - 7-9: Bordes gruesos (estilo infantil)
    
    **Threshold**
    - 50-80: Más bordes detectados (detallado)
    - 100: Equilibrado
    - 120-200: Menos bordes (minimalista)
    
    **Repeticiones de Filtro Bilateral**
    - 1-5: Suavizado ligero
    - 8-12: Equilibrado
    - 15+: Muy suavizado (colores muy planos)
    
    **Sigma Color**
    - Bajo (1-3): Solo píxeles muy similares se promedian
    - Alto (8-15): Más colores se mezclan → áreas más uniformes
    
    **Downsampling Factor**
    - 2: Alta calidad, lento
    - 4: Equilibrado (recomendado)
    - 6-8: Rápido, calidad reducida
    
    ### Aplicaciones Reales
    
    - **Aplicaciones móviles** - Filtros de cámara en tiempo real
    - **Videojuegos** - Estilización de gráficos
    - **Animación** - Pre-procesamiento para rotoscopia
    - **Arte digital** - Conversión foto-a-ilustración
    - **Publicación** - Ilustraciones para libros y revistas
    - **Efectos visuales** - Post-producción cinematográfica
    
    ### Tips para Mejores Resultados
    
    **Usa fotos con buena iluminación** - Contraste claro ayuda
    **Fondos simples funcionan mejor** - Menos distracción
    **Retratos son ideales** - Rostros se convierten bien
    **Ajusta threshold según contenido** - Fotos oscuras necesitan threshold menor
    **Más repeticiones para fotos ruidosas** - Suaviza imperfecciones
    
    **Evita:**
    - Fotos muy oscuras o subexpuestas
    - Fondos muy texturizados
    - Imágenes con mucho ruido
    - Resoluciones muy pequeñas
    
    ### Comparación con Otras Técnicas
    
    | Técnica | Bordes | Colores | Complejidad | Uso |
    |---------|--------|---------|-------------|-----|
    | **Cartoonización** | Gruesos y negros | Planos | Media | Cómics, animación |
    | **Oil Painting** | Suaves | Mezclados | Alta | Arte pintoresco |
    | **Pencil Sketch** | Finos y grises | N/A | Baja | Bocetos |
    | **Watercolor** | Difusos | Transparentes | Alta | Arte acuarela |
    
    ### Optimizaciones de Rendimiento
    
    El **downsampling** es clave para el rendimiento:
    
    - Imagen original: 1920×1080 = 2,073,600 píxeles
    - Con factor 4: 480×270 = 129,600 píxeles
    - **Reducción: 94% menos píxeles a procesar**
    
    El filtro bilateral es **computacionalmente costoso** (O(n² × r²) por píxel), 
    por eso aplicarlo a una imagen reducida acelera drásticamente el proceso.
    
    """)
    
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo = '''import cv2
import numpy as np

def cartoonize_image(img, ksize=5, sketch_mode=False):
    """Convierte una imagen en cartoon o sketch."""
    
    num_repetitions = 10  # Repeticiones del filtro bilateral
    sigma_color = 5       # Rango de color
    sigma_space = 7       # Rango espacial
    ds_factor = 4         # Factor de downsampling
    
    # 1. Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar median blur para reducir ruido
    img_gray = cv2.medianBlur(img_gray, 7)
    
    # 3. Detectar bordes con Laplacian
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    
    # 4. Umbralizar para obtener máscara binaria
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Si solo queremos el sketch, retornar la máscara
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 5. Reducir tamaño de la imagen (downsampling)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor,
                          interpolation=cv2.INTER_AREA)
    
    # 6. Aplicar filtro bilateral múltiples veces
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    
    # 7. Restaurar tamaño original (upsampling)
    img_output = cv2.resize(img_small, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    
    # 8. Combinar con máscara de bordes usando AND
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    
    return dst

# Usar la función
img = cv2.imread('foto.jpg')

# Modo cartoon a color
cartoon = cartoonize_image(img, ksize=5, sketch_mode=False)

# Modo sketch (solo bordes)
sketch = cartoonize_image(img, ksize=5, sketch_mode=True)

cv2.imshow('Cartoon', cartoon)
cv2.imshow('Sketch', sketch)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo)
    
    st.markdown("---")
    crear_seccion("Variaciones y Extensiones", "")
    
    st.markdown("""
    ### Ideas para Extender el Algoritmo
    
    1. **Cartoonización HDR**
       - Aplicar tone mapping antes de cartoonizar
       - Mejora contraste en fotos de alto rango dinámico
    
    2. **Cartoon con Paleta de Colores**
       - Cuantizar colores a una paleta específica (K-means)
       - Estilo más "animado" con colores limitados
    
    3. **Cartoon Adaptativo**
       - Ajustar parámetros según contenido de la imagen
       - Detectar rostros y aplicar diferentes parámetros
    
    4. **Multi-escala**
       - Detectar bordes en múltiples escalas
       - Capturar tanto detalles finos como formas grandes
    
    5. **Estilos Artísticos**
       - Combinar con transferencia de estilo neuronal
       - Simular estilos de artistas específicos
    """)


def cartoonize_image(img, ksize, sketch_mode, num_repetitions, 
                     sigma_color, sigma_space, ds_factor, threshold_val, median_blur):
    """
    Aplica el efecto de cartoonización a una imagen.
    
    Args:
        img: Imagen de entrada BGR
        ksize: Tamaño del kernel Laplacian
        sketch_mode: Si True, retorna solo el sketch
        num_repetitions: Número de veces que se aplica el filtro bilateral
        sigma_color: Sigma para el rango de color
        sigma_space: Sigma para el rango espacial
        ds_factor: Factor de downsampling
        threshold_val: Valor de umbral para detección de bordes
        median_blur: Tamaño del kernel median blur
        
    Returns:
        Imagen cartoonizada
    """
    # 1. Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar median blur
    img_gray = cv2.medianBlur(img_gray, median_blur)
    
    # 3. Detectar bordes con Laplacian
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    
    # 4. Umbralizar
    ret, mask = cv2.threshold(edges, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Si es modo sketch, retornar solo la máscara
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 5. Downsampling
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor,
                          interpolation=cv2.INTER_AREA)
    
    # 6. Aplicar filtro bilateral múltiples veces
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    
    # 7. Upsampling
    img_output = cv2.resize(img_small, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    
    # 8. Aplicar máscara
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    
    return dst


def cargar_imagen_input():
    """Carga imagen desde archivo o upload."""
    with st.sidebar:
        st.markdown("### Cargar Imagen")
        
        opcion = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key="img_source_cartoon"
        )
        
        if opcion == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen",
                key="upload_cartoon"
            )
            if archivo:
                return cargar_imagen_desde_upload(archivo)
            else:
                return None
        else:
            img_path = Path("data/images/input.jpg")
            if img_path.exists():
                return leer_imagen(str(img_path))
            else:
                # Crear imagen de ejemplo
                return crear_imagen_ejemplo()


def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo para demostración."""
    # Crear una imagen con formas geométricas y colores
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Fondo con gradiente
    for i in range(400):
        img[i, :] = [200 - i//3, 220 - i//4, 240 - i//5]
    
    # Círculo azul
    cv2.circle(img, (150, 150), 80, (255, 100, 50), -1)
    
    # Rectángulo rojo
    cv2.rectangle(img, (350, 80), (550, 250), (50, 50, 255), -1)
    
    # Triángulo verde
    pts = np.array([[300, 250], [200, 380], [400, 380]], np.int32)
    cv2.fillPoly(img, [pts], (50, 200, 50))
    
    # Texto
    cv2.putText(
        img,
        "CARTOON",
        (180, 350),
        cv2.FONT_HERSHEY_BOLD,
        1.5,
        (0, 0, 0),
        3
    )
    
    # Añadir algo de textura/ruido para que el filtro tenga efecto
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def guardar_resultado(img, nombre):
    """Guarda la imagen resultante."""
    from core.utils import guardar_imagen
    output_path = Path("data/output") / nombre
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if guardar_imagen(img, str(output_path)):
        st.success(f"Imagen guardada en: {output_path}")
    else:
        st.error("Error al guardar la imagen")