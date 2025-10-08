"""
Capítulo 5 - Ejercicio 5: Extracción de Características con SIFT
Aprende a detectar y visualizar keypoints usando SIFT (Scale-Invariant Feature Transform)
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
from ui.layout import crear_seccion, mostrar_codigo, crear_alerta
from ui.widgets import (
    control_slider,
    panel_control,
    checkbox_simple,
    selector_opciones,
    boton_accion,
    info_tooltip,
    entrada_numero
)

# Verificar disponibilidad de SIFT
try:
    # OpenCV >= 3.4.2
    sift_test = cv2.SIFT_create()
    SIFT_AVAILABLE = True
    SIFT_MODULE = "cv2.SIFT_create()"
except AttributeError:
    try:
        # OpenCV contrib (xfeatures2d)
        sift_test = cv2.xfeatures2d.SIFT_create()
        SIFT_AVAILABLE = True
        SIFT_MODULE = "cv2.xfeatures2d.SIFT_create()"
    except (AttributeError, cv2.error):
        SIFT_AVAILABLE = False
        SIFT_MODULE = None


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Extracción de Características SIFT")
    st.markdown("""
    Detecta y visualiza keypoints (puntos característicos) en imágenes usando **SIFT** 
    (Scale-Invariant Feature Transform), uno de los algoritmos más robustos para 
    detección de características invariantes a escala y rotación.
    """)
    
    st.markdown("---")
    
    # Verificar disponibilidad de SIFT
    if not verificar_sift():
        return
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Detección de Keypoints",
        "Análisis de Descriptores",
        "Comparación de Parámetros",
        "Teoría"
    ])
    
    with tab1:
        deteccion_keypoints()
    
    with tab2:
        analisis_descriptores()
    
    with tab3:
        comparacion_parametros()
    
    with tab4:
        mostrar_teoria()


def verificar_sift():
    """Verifica que SIFT esté disponible."""
    
    if not SIFT_AVAILABLE:
        st.error("SIFT no está disponible en tu instalación de OpenCV")
        
        st.markdown("""
        ### 🔧 Soluciones posibles:
        
        **Opción 1: Instalar OpenCV con soporte SIFT (Recomendado para uso educativo)**
        ```bash
        pip uninstall opencv-python opencv-contrib-python
        pip install opencv-contrib-python
        ```
        
        **Opción 2: Usar OpenCV >= 4.4.0**
        
        A partir de OpenCV 4.4.0, SIFT ya no está bajo patente y está disponible 
        en la versión estándar:
        ```bash
        pip install opencv-python>=4.4.0
        ```
        
        **Nota sobre la Patente:**
        
        SIFT estuvo patentado hasta marzo de 2020. Para **uso comercial** antes de esa fecha, 
        se requería licencia. Para **uso educativo y de investigación**, generalmente estaba permitido.
        
        **Alternativas libres de patentes:**
        - **ORB** (Oriented FAST and Rotated BRIEF) - Más rápido, libre
        - **AKAZE** - Precisión similar, libre
        - **BRISK** - Rápido y robusto, libre
        """)
        
        if st.button("Intentar nuevamente"):
            st.rerun()
        
        return False
    
    st.success(f"SIFT disponible: `{SIFT_MODULE}`")
    return True


def deteccion_keypoints():
    """Detección y visualización de keypoints."""
    
    crear_seccion("Detección de Keypoints", "")
    
    st.markdown("""
    Los **keypoints** son puntos de interés únicos en una imagen (esquinas, bordes, regiones distintivas) 
    que pueden ser detectados de manera confiable incluso cuando la imagen es escalada, rotada o 
    ligeramente deformada.
    """)
    
    # Sidebar para controles
    with st.sidebar:
        st.markdown("### Configuración")
        
        # Selector de imagen
        opcion_imagen = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo 1", "Imagen de ejemplo 2", "Subir imagen"],
            key="img_source_sift"
        )
        
        if opcion_imagen == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube una imagen",
                key="upload_sift"
            )
            if archivo:
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("Por favor sube una imagen")
                return
        else:
            # Usar imágenes de ejemplo
            img_paths = {
                "Imagen de ejemplo 1": "data/images/puente.jpg",
                "Imagen de ejemplo 2": "data/images/edificio.jpg"
            }
            img_path = Path(img_paths.get(opcion_imagen, img_paths["Imagen de ejemplo 1"]))
            
            if img_path.exists():
                img = leer_imagen(str(img_path))
            else:
                st.warning(f"No se encontró {img_path}. Por favor sube tu propia imagen.")
                return
        
        if img is None:
            return
        
        st.markdown("---")
        st.markdown("### Parámetros SIFT")
        
        # Parámetros SIFT
        n_features = entrada_numero(
            "Número de características",
            0, 10000, 0, 100,
            formato="%d",
            ayuda="0 = detectar todas las características posibles",
            key="n_features"
        )
        
        n_octave_layers = control_slider(
            "Capas por octava",
            1, 10, 3,
            "Número de capas en cada octava de la pirámide de escala",
            key="n_octave_layers"
        )
        
        contrast_threshold = st.sidebar.number_input(
            "Umbral de contraste",
            min_value=0.001,
            max_value=0.5,
            value=0.04,
            step=0.01,
            format="%.3f",
            help="Umbral para filtrar keypoints de bajo contraste",
            key="contrast_threshold"
        )
        
        edge_threshold = entrada_numero(
            "Umbral de borde",
            1.0, 50.0, 10.0, 1.0,
            formato="%.1f",
            ayuda="Umbral para filtrar keypoints en bordes",
            key="edge_threshold"
        )
        
        sigma = st.sidebar.number_input(
            "Sigma (σ)",
            min_value=0.5,
            max_value=5.0,
            value=1.6,
            step=0.1,
            format="%.1f",
            help="Sigma del Gaussian aplicado a la imagen base",
            key="sigma"
        )
        
        st.markdown("---")
        st.markdown("### Opciones de Visualización")
        
        draw_mode = selector_opciones(
            "Modo de dibujo",
            ["Keypoints ricos", "Keypoints simples", "Solo ubicaciones"],
            key="draw_mode"
        )
        
        color_mode = selector_opciones(
            "Color de keypoints",
            ["Aleatorio", "Verde", "Azul", "Rojo", "Por escala"],
            key="color_mode"
        )
        
        mostrar_info = checkbox_simple(
            "Mostrar información detallada",
            True,
            key="show_info"
        )
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crear detector SIFT
    sift = crear_sift_detector(
        n_features,
        n_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma
    )
    
    # Detectar keypoints
    with st.spinner("Detectando keypoints..."):
        keypoints = sift.detect(gray, None)
    
    # Dibujar keypoints
    img_keypoints = img.copy()
    img_keypoints = dibujar_keypoints(
        img_keypoints,
        keypoints,
        draw_mode,
        color_mode
    )
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img, "")
    
    with col2:
        st.markdown(f"**Keypoints Detectados ({len(keypoints)})**")
        mostrar_imagen_streamlit(img_keypoints, "")
    
    # Información detallada
    if mostrar_info and len(keypoints) > 0:
        mostrar_informacion_keypoints(keypoints)
    
    # Opciones de análisis adicional
    st.markdown("---")
    crear_seccion("Análisis Adicional", "")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Ver distribución de escalas", use_container_width=True):
            mostrar_distribucion_escalas(keypoints)
    
    with col2:
        if st.button("Ver distribución de orientaciones", use_container_width=True):
            mostrar_distribucion_orientaciones(keypoints)
    
    with col3:
        if st.button("Ver mapa de calor", use_container_width=True):
            mostrar_mapa_calor(img, keypoints)
    
    # Botón de descarga
    if len(keypoints) > 0 and boton_accion("Guardar resultado", key="save_sift"):
        guardar_resultado(img_keypoints, "sift_keypoints.jpg")


def analisis_descriptores():
    """Análisis de descriptores SIFT."""
    
    crear_seccion("Análisis de Descriptores", "")
    
    st.markdown("""
    Los **descriptores** son vectores numéricos que describen la región alrededor de cada keypoint.
    SIFT genera descriptores de **128 dimensiones** que son robustos a cambios de iluminación,
    escala y rotación.
    """)
    
    # Cargar imagen
    with st.sidebar:
        st.markdown("### Configuración")
        
        opcion_imagen = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo 1", "Imagen de ejemplo 2", "Subir imagen"],
            key="img_source_desc"
        )
        
        if opcion_imagen == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube una imagen",
                key="upload_desc"
            )
            if archivo:
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("Por favor sube una imagen")
                return
        else:
            img_paths = {
                "Imagen de ejemplo 1": "data/images/puente.jpg",
                "Imagen de ejemplo 2": "data/images/edificio.jpg"
            }
            img_path = Path(img_paths.get(opcion_imagen, img_paths["Imagen de ejemplo 1"]))
            
            if img_path.exists():
                img = leer_imagen(str(img_path))
            else:
                st.warning(f"No se encontró la imagen de ejemplo")
                return
        
        if img is None:
            return
        
        max_keypoints_display = control_slider(
            "Keypoints a mostrar",
            5, 50, 10,
            "Número de keypoints para visualizar descriptores",
            key="max_kp_display"
        )
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crear detector SIFT y calcular descriptores
    sift = crear_sift_detector()
    
    with st.spinner("Calculando keypoints y descriptores..."):
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        st.warning("No se detectaron keypoints en esta imagen")
        return
    
    # Mostrar estadísticas generales
    st.markdown("### Estadísticas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Keypoints", len(keypoints))
    
    with col2:
        st.metric("Dimensiones", descriptors.shape[1])
    
    with col3:
        avg_response = np.mean([kp.response for kp in keypoints])
        st.metric("Respuesta promedio", f"{avg_response:.3f}")
    
    with col4:
        avg_size = np.mean([kp.size for kp in keypoints])
        st.metric("Tamaño promedio", f"{avg_size:.1f}px")
    
    st.markdown("---")
    
    # Visualizar algunos descriptores
    st.markdown("### Visualización de Descriptores")
    
    st.info(f"Mostrando los primeros {min(max_keypoints_display, len(keypoints))} keypoints más fuertes")
    
    # Ordenar keypoints por respuesta
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
    keypoints_top = keypoints_sorted[:max_keypoints_display]
    indices_top = [keypoints.index(kp) for kp in keypoints_top]
    
    # Dibujar keypoints seleccionados
    img_selected = img.copy()
    for i, kp in enumerate(keypoints_top):
        color = (0, 255, 0) if i == 0 else (255, 0, 0)
        cv2.circle(img_selected, (int(kp.pt[0]), int(kp.pt[1])), 
                  int(kp.size), color, 2)
        cv2.putText(img_selected, str(i+1), 
                   (int(kp.pt[0])-10, int(kp.pt[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    mostrar_imagen_streamlit(img_selected, f"Keypoints seleccionados (Verde = más fuerte)")
    
    st.markdown("---")
    
    # Análisis estadístico de descriptores
    st.markdown("### Análisis Estadístico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estadísticas de Descriptores**")
        st.code(f"""
Media global: {np.mean(descriptors):.2f}
Desviación estándar: {np.std(descriptors):.2f}
Valor mínimo: {np.min(descriptors):.2f}
Valor máximo: {np.max(descriptors):.2f}
Mediana: {np.median(descriptors):.2f}
        """)
    
    with col2:
        st.markdown("**Propiedades del Mejor Keypoint**")
        best_kp = keypoints_top[0]
        st.code(f"""
Posición: ({int(best_kp.pt[0])}, {int(best_kp.pt[1])})
Tamaño: {best_kp.size:.2f}px
Ángulo: {best_kp.angle:.2f}°
Respuesta: {best_kp.response:.4f}
Octava: {best_kp.octave}
        """)


def comparacion_parametros():
    """Comparación de diferentes configuraciones de parámetros."""
    
    crear_seccion("Comparación de Parámetros", "")
    
    st.markdown("""
    Observa cómo diferentes parámetros de SIFT afectan la cantidad y calidad 
    de los keypoints detectados.
    """)
    
    # Cargar imagen
    img_path = Path("data/images/puente.jpg")
    if not img_path.exists():
        img_path = Path("data/images/edificio.jpg")
    
    if img_path.exists():
        img = leer_imagen(str(img_path))
    else:
        st.warning("No se encontró imagen de ejemplo")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    st.markdown("### Efecto del Umbral de Contraste")
    
    contrast_values = [0.02, 0.04, 0.08, 0.12]
    
    cols = st.columns(len(contrast_values))
    
    for i, contrast in enumerate(contrast_values):
        with cols[i]:
            sift = crear_sift_detector(contrast_threshold=contrast)
            keypoints = sift.detect(gray, None)
            
            img_temp = img.copy()
            img_temp = dibujar_keypoints(img_temp, keypoints, "Keypoints ricos", "Verde")
            
            st.markdown(f"**Contraste: {contrast}**")
            st.caption(f"{len(keypoints)} keypoints")
            mostrar_imagen_streamlit(img_temp, "")
    
    st.markdown("---")
    st.markdown("### Efecto de las Capas por Octava")
    
    octave_values = [2, 3, 4, 5]
    
    cols = st.columns(len(octave_values))
    
    for i, octaves in enumerate(octave_values):
        with cols[i]:
            sift = crear_sift_detector(n_octave_layers=octaves)
            keypoints = sift.detect(gray, None)
            
            img_temp = img.copy()
            img_temp = dibujar_keypoints(img_temp, keypoints, "Keypoints ricos", "Azul")
            
            st.markdown(f"**Octavas: {octaves}**")
            st.caption(f"{len(keypoints)} keypoints")
            mostrar_imagen_streamlit(img_temp, "")
    
    st.markdown("---")
    
    # Tabla comparativa
    st.markdown("### Tabla Comparativa")
    
    configs = [
        ("Muy permisivo", 0, 3, 0.02, 10, 1.6),
        ("Estándar", 0, 3, 0.04, 10, 1.6),
        ("Restrictivo", 0, 3, 0.08, 15, 1.6),
        ("Muy restrictivo", 0, 3, 0.12, 20, 1.6),
    ]
    
    results = []
    
    for nombre, nf, no, ct, et, s in configs:
        sift = crear_sift_detector(nf, no, ct, et, s)
        keypoints = sift.detect(gray, None)
        avg_response = np.mean([kp.response for kp in keypoints]) if keypoints else 0
        avg_size = np.mean([kp.size for kp in keypoints]) if keypoints else 0
        
        results.append({
            "Configuración": nombre,
            "Keypoints": len(keypoints),
            "Respuesta Promedio": f"{avg_response:.4f}",
            "Tamaño Promedio": f"{avg_size:.1f}px",
            "Umbral Contraste": ct,
            "Umbral Borde": et
        })
    
    import pandas as pd
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)
    
    st.info("""
    **Observaciones:**
    - Umbrales más bajos → Más keypoints (incluye puntos débiles)
    - Umbrales más altos → Menos keypoints (solo los más robustos)
    - Más capas por octava → Mejor detección multi-escala pero más lento
    """)


def mostrar_teoria():
    """Sección teórica sobre SIFT."""
    
    crear_seccion("Teoría: Scale-Invariant Feature Transform (SIFT)", "")
    
    st.markdown("""
    ### ¿Qué es SIFT?
    
    **SIFT** (Scale-Invariant Feature Transform) es un algoritmo de visión por computadora 
    desarrollado por **David Lowe en 1999** (publicado en 2004) para detectar y describir 
    características locales en imágenes.
    
    ### Características Principales
    
    #### **Invarianzas:**
    - **Escala**: Detecta características independientemente del tamaño
    - **Rotación**: No afectado por la orientación de la imagen
    - **Iluminación**: Robusto a cambios de brillo y contraste
    - **Punto de vista**: Parcialmente invariante a cambios de perspectiva
    - **Ruido**: Resistente al ruido en la imagen
    
    ### Pipeline de SIFT (4 Etapas)
    
    #### **1. Detección de Extremos en Espacio-Escala**
    
    ```
    Pirámide Gaussian → Diferencia de Gaussianas (DoG) → Extremos locales
    ```
    
    - Crea pirámide de imágenes en diferentes escalas
    - Aplica Diferencia de Gaussianas (DoG): aproximación del Laplaciano
    - Busca máximos y mínimos locales en el espacio 3D (x, y, escala)
    
    #### **2. Localización Precisa de Keypoints**
    
    - Elimina puntos de bajo contraste (inestables)
    - Elimina puntos en bordes (usando matriz Hessiana)
    - Refina la posición mediante interpolación sub-pixel
    
    #### **3. Asignación de Orientación**
    
    - Calcula histograma de orientaciones en vecindad del keypoint
    - Asigna orientación(es) dominante(s)
    - Hace al descriptor invariante a rotación
    
    #### **4. Generación de Descriptores**
    
    - Región de 16x16 píxeles alrededor del keypoint
    - Dividida en 4x4 sub-regiones
    - Histograma de 8 orientaciones por sub-región
    - **Resultado: Vector de 128 dimensiones (4×4×8)**
    
    ### Parámetros de SIFT
    
    ```python
    sift = cv2.SIFT_create(
        nfeatures=0,           # 0 = ilimitado
        nOctaveLayers=3,       # Capas por octava
        contrastThreshold=0.04, # Umbral de contraste
        edgeThreshold=10,      # Umbral de borde
        sigma=1.6              # Sigma del Gaussian
    )
    ```
    
    #### **nfeatures**
    - Número máximo de características a retener
    - 0 = detectar todas las posibles
    - Útil para limitar tiempo de procesamiento
    
    #### **nOctaveLayers**
    - Número de capas por octava en pirámide DoG
    - Más capas = mejor detección multi-escala
    - Valor típico: 3
    - Rango común: 2-5
    
    #### **contrastThreshold**
    - Filtra keypoints de bajo contraste
    - Mayor valor = menos keypoints, más robustos
    - Valor típico: 0.04
    - Rango: 0.01 - 0.15
    
    #### **edgeThreshold**
    - Filtra keypoints en bordes (ratio Hessiana)
    - Mayor valor = menos rechazo de bordes
    - Valor típico: 10
    - Rango: 5 - 20
    
    #### **sigma**
    - Desviación estándar del Gaussian base
    - Controla suavizado inicial
    - Valor típico: 1.6
    - Rango: 1.0 - 2.0
    
    ### Propiedades de un Keypoint
    
    Cada keypoint detectado contiene:
    
    ```python
    keypoint.pt        # (x, y) coordenadas
    keypoint.size      # Diámetro del área significativa
    keypoint.angle     # Orientación en grados [0, 360)
    keypoint.response  # Fuerza de la respuesta (calidad)
    keypoint.octave    # Octava donde fue detectado
    keypoint.class_id  # ID de clase (para clasificación)
    ```
    
    ### Descriptor SIFT (128D)
    
    El descriptor es un vector de 128 valores que describe la región:
    
    ```
    [v₀, v₁, v₂, ..., v₁₂₇]
    
    Organizado como: 4×4 sub-regiones × 8 orientaciones = 128
    ```
    
    **Propiedades:**
    - Normalizado para invarianza a iluminación
    - Robusto a deformaciones geométricas pequeñas
    - Único y distintivo para matching
    
    ### Diferencia de Gaussianas (DoG)
    
    ```
    DoG(x, y, σ) = G(x, y, kσ) - G(x, y, σ)
    ```
    
    Donde:
    - G = Filtro Gaussiano
    - k = Factor de escala constante
    - σ = Desviación estándar
    
    **DoG aproxima la Laplaciana normalizada de Gaussiana**, que es óptima 
    para detección de blobs en espacio-escala.
    
    ### Ventajas de SIFT
    
    **Muy robusto** a transformaciones
    **Descriptores distintivos** - excelente para matching
    **Bien establecido** - ampliamente investigado y validado
    **Implementación madura** en OpenCV
    **Funciona en escenas complejas**
    
    ### Desventajas de SIFT
    
    **Computacionalmente costoso** comparado con alternativas modernas
    **No funciona bien** en imágenes con repetición de patrones
    **Descriptores grandes** (128D) - más memoria
    **Patente** (expiró en marzo 2020, ahora libre)
    
    ### Alternativas a SIFT
    
    | Algoritmo | Velocidad | Precisión | Descriptores | Libre |
    |-----------|-----------|-----------|--------------|-------|
    | **SIFT** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 128D | ✅ (desde 2020) |
    | **SURF** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 64D/128D | ❌ (patentado) |
    | **ORB** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 256 bits | ✅ |
    | **AKAZE** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 61D-486D | ✅ |
    | **BRISK** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 512 bits | ✅ |
    
    ### Aplicaciones de SIFT
    
    - **Reconocimiento de objetos** - Identificar objetos en escenas
    - **Panoramas** - Crear imágenes panorámicas (image stitching)
    - **SLAM** (Simultaneous Localization and Mapping) - Robots y drones
    - **Reconocimiento facial** - Matching de rostros
    - **Tracking** - Seguimiento de objetos en video
    - **Reconstrucción 3D** - Structure from Motion
    - **Watermarking** - Marcas de agua robustas
    - **Realidad Aumentada** - Detectar y trackear marcadores
    
    ### Tips para Mejores Resultados
    
    **Textura rica** - SIFT funciona mejor con texturas detalladas
    **Buenos bordes** - Esquinas y bordes fuertes mejoran detección
    **Evitar superficies planas** - Pocas características detectables
    **Buena iluminación** - Mejora la calidad de los descriptores
    **Imágenes nítidas** - El blur reduce la cantidad de keypoints
    
    ### Optimización de Rendimiento
    
    ```python
    # Para aplicaciones en tiempo real:
    sift = cv2.SIFT_create(
        nfeatures=500,          # Limitar número de características
        nOctaveLayers=3,        # No aumentar sin necesidad
        contrastThreshold=0.06  # Aumentar para menos keypoints
    )
    
    # Para matching preciso:
    sift = cv2.SIFT_create(
        nfeatures=0,            # Sin límite
        nOctaveLayers=4,        # Más escalas
        contrastThreshold=0.03  # Más permisivo
    )
    ```
    
    ### Comparación: SIFT vs Detección de Esquinas
    
    | Aspecto | Harris Corners | SIFT |
    |---------|---------------|------|
    | **Tipo** | Solo detección | Detección + Descripción |
    | **Escala** | ❌ No invariante | ✅ Invariante |
    | **Rotación** | ❌ No invariante | ✅ Invariante |
    | **Descriptores** | ❌ No incluye | ✅ 128D vector |
    | **Velocidad** | ⚡⚡⚡⚡⚡ | ⚡⚡ |
    | **Matching** | Difícil | Fácil |
    
    ### Historia y Patente
    
    - **1999**: David Lowe presenta SIFT
    - **2004**: Publicación completa del paper
    - **2004-2020**: Bajo patente (uso comercial restringido)
    - **Marzo 2020**: Patente expira - ahora completamente libre
    - **OpenCV 4.4.0+**: SIFT incluido en versión estándar
    
    ### Paper Original
    
    **"Distinctive Image Features from Scale-Invariant Keypoints"**
    David G. Lowe, 2004
    
    *Uno de los papers más citados en Computer Vision (>50,000 citaciones)*
    """)
    
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo = '''import cv2
import numpy as np

# Cargar imagen
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear detector SIFT
sift = cv2.SIFT_create()

# Opción 1: Solo detectar keypoints
keypoints = sift.detect(gray, None)

# Opción 2: Detectar y computar descriptores
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Dibujar keypoints
img_keypoints = cv2.drawKeypoints(
    img, 
    keypoints, 
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Mostrar información
print(f"Keypoints detectados: {len(keypoints)}")
print(f"Dimensión descriptores: {descriptors.shape}")

# Propiedades del primer keypoint
kp = keypoints[0]
print(f"Posición: {kp.pt}")
print(f"Tamaño: {kp.size}")
print(f"Ángulo: {kp.angle}")
print(f"Respuesta: {kp.response}")

# Visualizar
cv2.imshow('SIFT Features', img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    
    mostrar_codigo(codigo, "python")
    
    st.markdown("---")
    
    # Ejemplo de matching
    crear_seccion("Ejemplo: Matching de Keypoints", "")
    
    codigo_matching = '''import cv2
import numpy as np

# Cargar dos imágenes
img1 = cv2.imread('image1.jpg', 0)
img2 = cv2.imread('image2.jpg', 0)

# Detectar y computar
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matcher usando FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Ratio test (Lowe's ratio)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Dibujar matches
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

print(f"Matches encontrados: {len(good_matches)}")
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo_matching, "python")


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_sift_detector(n_features=0, n_octave_layers=3, 
                       contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
    """Crea un detector SIFT con los parámetros especificados."""
    try:
        # OpenCV >= 4.4.0
        sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    except AttributeError:
        # OpenCV con contrib
        sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    return sift


def dibujar_keypoints(img, keypoints, draw_mode, color_mode):
    """Dibuja keypoints en la imagen según el modo especificado."""
    
    # Determinar flags
    if draw_mode == "Keypoints ricos":
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    elif draw_mode == "Keypoints simples":
        flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT
    else:  # Solo ubicaciones
        flags = 0
    
    # Determinar color
    if color_mode == "Aleatorio":
        color = None
    elif color_mode == "Verde":
        color = (0, 255, 0)
    elif color_mode == "Azul":
        color = (255, 0, 0)
    elif color_mode == "Rojo":
        color = (0, 0, 255)
    elif color_mode == "Por escala":
        # Colorear según tamaño (escala)
        img_result = img.copy()
        for kp in keypoints:
            # Mapear tamaño a color (azul=pequeño, rojo=grande)
            size_normalized = min(kp.size / 50.0, 1.0)
            color_kp = (
                int(255 * size_normalized),
                0,
                int(255 * (1 - size_normalized))
            )
            cv2.circle(img_result, (int(kp.pt[0]), int(kp.pt[1])), 
                      int(kp.size), color_kp, 2)
            
            # Dibujar orientación
            if draw_mode == "Keypoints ricos":
                angle_rad = np.deg2rad(kp.angle)
                x2 = int(kp.pt[0] + kp.size * np.cos(angle_rad))
                y2 = int(kp.pt[1] + kp.size * np.sin(angle_rad))
                cv2.line(img_result, (int(kp.pt[0]), int(kp.pt[1])), 
                        (x2, y2), color_kp, 2)
        
        return img_result
    else:
        color = None
    
    # Dibujar
    img_result = cv2.drawKeypoints(img, keypoints, None, color, flags)
    
    return img_result


def mostrar_informacion_keypoints(keypoints):
    """Muestra información detallada sobre los keypoints."""
    
    st.markdown("### Información Detallada")
    
    # Estadísticas generales
    sizes = [kp.size for kp in keypoints]
    responses = [kp.response for kp in keypoints]
    angles = [kp.angle for kp in keypoints]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Tamaños**")
        st.code(f"""
Promedio: {np.mean(sizes):.2f}px
Mínimo: {np.min(sizes):.2f}px
Máximo: {np.max(sizes):.2f}px
Std Dev: {np.std(sizes):.2f}px
        """)
    
    with col2:
        st.markdown("**Respuestas**")
        st.code(f"""
Promedio: {np.mean(responses):.4f}
Mínimo: {np.min(responses):.4f}
Máximo: {np.max(responses):.4f}
Std Dev: {np.std(responses):.4f}
        """)
    
    with col3:
        st.markdown("**Orientaciones**")
        st.code(f"""
Promedio: {np.mean(angles):.2f}°
Mínimo: {np.min(angles):.2f}°
Máximo: {np.max(angles):.2f}°
Std Dev: {np.std(angles):.2f}°
        """)
    
    # Top 5 keypoints más fuertes
    st.markdown("---")
    st.markdown("### Top 5 Keypoints Más Fuertes")
    
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)[:5]
    
    for i, kp in enumerate(keypoints_sorted, 1):
        st.markdown(f"""
**#{i}** - Posición: ({int(kp.pt[0])}, {int(kp.pt[1])}) | 
Tamaño: {kp.size:.1f}px | Ángulo: {kp.angle:.1f}° | 
Respuesta: {kp.response:.4f}
        """)


def mostrar_distribucion_escalas(keypoints):
    """Muestra la distribución de escalas de los keypoints."""
    
    sizes = [kp.size for kp in keypoints]
    
    st.markdown("### Distribución de Escalas")
    
    # Crear histograma
    hist, bins = np.histogram(sizes, bins=20)
    
    # Mostrar con Streamlit
    import pandas as pd
    df = pd.DataFrame({
        'Tamaño (px)': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
        'Frecuencia': hist
    })
    
    st.bar_chart(df.set_index('Tamaño (px)'))
    
    st.info(f"""
    **Análisis:**
    - Rango de tamaños: {min(sizes):.1f}px - {max(sizes):.1f}px
    - Tamaño más común: {bins[np.argmax(hist)]:.1f}px
    - Total de keypoints: {len(keypoints)}
    """)


def mostrar_distribucion_orientaciones(keypoints):
    """Muestra la distribución de orientaciones de los keypoints."""
    
    angles = [kp.angle for kp in keypoints]
    
    st.markdown("### Distribución de Orientaciones")
    
    # Crear histograma circular (simplificado)
    hist, bins = np.histogram(angles, bins=36)  # 10° por bin
    
    import pandas as pd
    df = pd.DataFrame({
        'Ángulo (°)': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)],
        'Frecuencia': hist
    })
    
    st.bar_chart(df.set_index('Ángulo (°)'))
    
    st.info(f"""
    **Análisis:**
    - Orientación promedio: {np.mean(angles):.1f}°
    - Desviación estándar: {np.std(angles):.1f}°
    - Total de keypoints: {len(keypoints)}
    """)


def mostrar_mapa_calor(img, keypoints):
    """Muestra un mapa de calor de la densidad de keypoints."""
    
    st.markdown("### Mapa de Calor de Keypoints")
    
    # Crear imagen de densidad
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        response = kp.response
        
        # Agregar Gaussian blob
        y_min = max(0, y - size)
        y_max = min(heatmap.shape[0], y + size)
        x_min = max(0, x - size)
        x_max = min(heatmap.shape[1], x + size)
        
        heatmap[y_min:y_max, x_min:x_max] += response
    
    # Normalizar
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    
    # Aplicar colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer
    overlay = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Mapa de Calor Puro**")
        mostrar_imagen_streamlit(heatmap_colored, "")
    
    with col2:
        st.markdown("**Superposición**")
        mostrar_imagen_streamlit(overlay, "")
    
    st.info("""
    **Interpretación:**
    - Rojo/Amarillo = Alta densidad de keypoints fuertes
    - Verde/Azul = Baja densidad de keypoints
    - Negro = Sin keypoints
    """)


def guardar_resultado(img, filename):
    """Guarda la imagen resultado."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), img)
    
    st.success(f"Imagen guardada en: {output_path}")


if __name__ == "__main__":
    run()