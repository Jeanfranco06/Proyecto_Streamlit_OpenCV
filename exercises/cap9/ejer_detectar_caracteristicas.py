"""
Capítulo 9 - Ejercicio 9: Detectores de Características (Features)
Aprende a detectar puntos clave en imágenes usando diferentes métodos:
- Dense Detector (detección en grilla uniforme)
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- Otros detectores modernos
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from core.utils import (
    leer_imagen,
    bgr_to_rgb,
    mostrar_imagen_streamlit,
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


# ==================== DETECTORES ====================

class DenseDetector():
    """
    Detector de características denso.
    Crea keypoints en una grilla uniforme sobre la imagen.
    """
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        """
        Args:
            step_size: Tamaño del descriptor del keypoint
            feature_scale: Espaciado entre keypoints
            img_bound: Margen desde los bordes de la imagen
        """
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
    
    def detect(self, img):
        """Detecta keypoints en grilla uniforme."""
        keypoints = []
        rows, cols = img.shape[:2]
        
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(y), float(x), self.initXyStep))
        
        return keypoints


class SIFTDetector():
    """
    Detector SIFT (Scale-Invariant Feature Transform).
    Detecta características invariantes a escala y rotación.
    """
    def __init__(self):
        try:
            # Intentar con xfeatures2d (OpenCV contrib)
            self.detector = cv2.xfeatures2d.SIFT_create()
        except AttributeError:
            # Usar SIFT del módulo principal (OpenCV 4.4+)
            self.detector = cv2.SIFT_create()
    
    def detect(self, img):
        """Detecta keypoints usando SIFT."""
        # Convertir a escala de grises
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar keypoints
        return self.detector.detect(gray_image, None)


class ORBDetector():
    """
    Detector ORB (Oriented FAST and Rotated BRIEF).
    Alternativa rápida y libre de patentes a SIFT.
    """
    def __init__(self, n_features=500):
        self.detector = cv2.ORB_create(nfeatures=n_features)
    
    def detect(self, img):
        """Detecta keypoints usando ORB."""
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)


class BRISKDetector():
    """
    Detector BRISK (Binary Robust Invariant Scalable Keypoints).
    Detector rápido con descriptores binarios.
    """
    def __init__(self):
        self.detector = cv2.BRISK_create()
    
    def detect(self, img):
        """Detecta keypoints usando BRISK."""
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)


class AKAZEDetector():
    """
    Detector AKAZE (Accelerated-KAZE).
    Detector moderno con buen balance entre velocidad y calidad.
    """
    def __init__(self):
        self.detector = cv2.AKAZE_create()
    
    def detect(self, img):
        """Detecta keypoints usando AKAZE."""
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)


# ==================== APLICACIÓN STREAMLIT ====================

def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Detectores de Características (Features)")
    st.markdown("""
    Los **detectores de características** identifican puntos de interés (keypoints) en imágenes.
    Estos puntos son fundamentales para reconocimiento de objetos, tracking, panoramas, 
    realidad aumentada y más.
    """)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Comparación de Detectores",
        "Análisis de Keypoints",
        "Visualización Detallada",
        "Teoría"
    ])
    
    with tab1:
        comparacion_detectores()
    
    with tab2:
        analisis_keypoints()
    
    with tab3:
        visualizacion_detallada()
    
    with tab4:
        mostrar_teoria()


def comparacion_detectores():
    """Compara diferentes detectores de características."""
    
    crear_seccion("Comparación de Detectores", "")
    
    st.markdown("""
    Compara el rendimiento y resultados de diferentes algoritmos de detección de características.
    """)
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_comparacion")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Selector de detectores a comparar
    with panel_control("Selecciona Detectores"):
        detectores_disponibles = ["Dense", "SIFT", "ORB", "BRISK", "AKAZE"]
        
        detectores_seleccionados = st.multiselect(
            "Detectores a comparar",
            detectores_disponibles,
            default=["Dense", "SIFT", "ORB"],
            help="Selecciona 2 o más detectores"
        )
        
        if len(detectores_seleccionados) < 1:
            st.warning("Selecciona al menos un detector")
            return
        
        # Parámetros generales
        mostrar_info = checkbox_simple(
            "Mostrar información de keypoints",
            True,
            key="show_kp_info"
        )
    
    # Detectar y mostrar
    st.markdown("---")
    crear_seccion("Resultados", "")
    
    resultados = {}
    
    # Configuraciones por defecto para cada detector
    configs = {
        "Dense": {"step_size": 20, "feature_scale": 20, "img_bound": 5},
        "SIFT": {},
        "ORB": {"n_features": 500},
        "BRISK": {},
        "AKAZE": {}
    }
    
    # Procesar cada detector
    for detector_name in detectores_seleccionados:
        with st.spinner(f"Procesando {detector_name}..."):
            keypoints, img_result, tiempo = detectar_keypoints(
                img, 
                detector_name, 
                configs[detector_name]
            )
            resultados[detector_name] = {
                "keypoints": keypoints,
                "imagen": img_result,
                "tiempo": tiempo,
                "cantidad": len(keypoints)
            }
    
    # Mostrar resultados en grid
    num_cols = min(3, len(detectores_seleccionados))
    
    for i in range(0, len(detectores_seleccionados), num_cols):
        cols = st.columns(num_cols)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(detectores_seleccionados):
                detector_name = detectores_seleccionados[idx]
                resultado = resultados[detector_name]
                
                with col:
                    st.markdown(f"**{detector_name}**")
                    mostrar_imagen_streamlit(
                        resultado["imagen"], 
                        "", 
                        use_column_width=True
                    )
                    
                    if mostrar_info:
                        st.metric(
                            "Keypoints", 
                            f"{resultado['cantidad']:,}",
                            help="Número de características detectadas"
                        )
                        st.caption(f"{resultado['tiempo']:.3f}s")


def analisis_keypoints():
    """Análisis estadístico de los keypoints detectados."""
    
    crear_seccion("Análisis Estadístico de Keypoints", "")
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_analisis")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Selector de detector
    detector_name = selector_opciones(
        "Detector",
        ["SIFT", "ORB", "BRISK", "AKAZE"],
        key="detector_analisis"
    )
    
    # Detectar keypoints
    keypoints, img_result, tiempo = detectar_keypoints(img, detector_name, {})
    
    if len(keypoints) == 0:
        st.warning("No se detectaron keypoints en la imagen")
        return
    
    # Extraer propiedades
    positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    sizes = np.array([kp.size for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    angles = np.array([kp.angle for kp in keypoints])
    
    # Visualización general
    st.markdown("---")
    crear_seccion("Distribución de Keypoints", "")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen con Keypoints**")
        mostrar_imagen_streamlit(img_result, "")
    
    with col2:
        st.markdown("**Mapa de Calor**")
        heatmap = crear_heatmap_keypoints(img, positions)
        mostrar_imagen_streamlit(heatmap, "")
    


def visualizacion_detallada():
    """Visualización detallada de keypoints individuales."""
    
    crear_seccion("Explorador de Keypoints", "")
    
    st.markdown("""
    Explora keypoints individuales y sus propiedades en detalle.
    """)
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_viz")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Detectar keypoints
    detector_name = selector_opciones(
        "Detector",
        ["SIFT", "ORB", "BRISK", "AKAZE"],
        key="detector_viz"
    )
    
    keypoints, img_result, tiempo = detectar_keypoints(img, detector_name, {})
    
    if len(keypoints) == 0:
        st.warning("No se detectaron keypoints")
        return
    
    st.info(f"Detectados {len(keypoints)} keypoints en {tiempo:.3f}s")
    
    # Mostrar imagen general
    st.markdown("---")
    mostrar_imagen_streamlit(img_result, "Vista General")
    
    # Filtros
    st.markdown("---")
    crear_seccion("Filtrar Keypoints", "")
    
    with panel_control("Opciones de Filtrado"):
        
        # Top N keypoints por response
        top_n = control_slider(
            "Mostrar top N keypoints (por response)",
            10, min(200, len(keypoints)), 
            min(50, len(keypoints)),
            key="top_n"
        )
        
        # Ordenar por response y tomar top N
        keypoints_sorted = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        keypoints_filtered = keypoints_sorted[:top_n]
        
        # Redibujar con keypoints filtrados
        img_filtered = np.copy(img)
        img_filtered = cv2.drawKeypoints(
            img_filtered, 
            keypoints_filtered, 
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    
    st.markdown("---")
    mostrar_imagen_streamlit(img_filtered, f"Top {top_n} Keypoints (por response)")


def mostrar_teoria():
    """Explicación teórica de los detectores de características."""
    
    crear_seccion("Teoría: Detectores de Características", "")
    
    st.markdown("""
    ### ¿Qué son las Características (Features)?
    
    Las **características** o **features** son puntos distintivos en una imagen que son:
    - **Fáciles de localizar** en diferentes condiciones
    - **Robustos** a cambios de escala, rotación, iluminación
    - **Únicos** y distinguibles de otros puntos
    - **Repetibles** - se detectan consistentemente
    
    ### Keypoints (Puntos Clave)
    
    Un **keypoint** es un punto de interés detectado en una imagen. Cada keypoint tiene:
    
    | Propiedad | Descripción |
    |-----------|-------------|
    | **pt (x, y)** | Posición en píxeles |
    | **size** | Tamaño/escala del descriptor |
    | **angle** | Orientación (0-360°, -1 si no definido) |
    | **response** | Fuerza/calidad de la detección |
    | **octave** | Nivel de pirámide donde se detectó |
    | **class_id** | Identificador de clase (uso específico) |
    
    ### Tipos de Detectores
    
    ## 1. Dense Detector (Detector Denso)
    
    **Concepto**: No es un "detector" real - simplemente coloca keypoints en una **grilla uniforme**.
    
    #### Características:
    - **Muy rápido** - solo calcula posiciones geométricas
    - **Predecible** - siempre el mismo número de keypoints
    - **Cobertura uniforme** de toda la imagen
    - **No discrimina** entre regiones interesantes y aburridas
    - **No invariante** a transformaciones
    
    #### Parámetros:
    - **step_size**: Tamaño del descriptor del keypoint
    - **feature_scale**: Espaciado entre keypoints (en píxeles)
    - **img_bound**: Margen desde los bordes
    
    #### Uso típico:
    ```python
    detector = DenseDetector(step_size=20, feature_scale=20, img_bound=5)
    keypoints = detector.detect(img)
    ```
    
    #### Cuándo usar:
    - Cuando necesitas **cobertura completa** de la imagen
    - Para **bag-of-features** o descriptores densos
    - Como baseline para comparaciones
    - Cuando la velocidad es crítica
    
    ---
    
    ## 2. SIFT (Scale-Invariant Feature Transform)
    
    **Concepto**: Detecta características **invariantes a escala y rotación** usando diferencias de Gaussianas (DoG).
    
    #### Características:
    - **Muy robusto** a escala, rotación, iluminación
    - **Alta calidad** de detección
    - **Descriptores distintivos** (128 dimensiones)
    - **Lento** - computacionalmente costoso
    - **Patentado** (libre desde 2020)
    
    #### Proceso (simplificado):
    1. Construye pirámide de escalas con Gaussian blur
    2. Calcula Diferencia de Gaussianas (DoG)
    3. Encuentra extremos locales en espacio escala
    4. Refina ubicación sub-píxel
    5. Elimina keypoints de baja respuesta o en bordes
    6. Asigna orientación dominante
    7. Genera descriptor de 128 dimensiones
    
    #### Ventajas clave:
    - **Invariante a escala**: Detecta objetos a diferentes tamaños
    - **Invariante a rotación**: El ángulo se calcula automáticamente
    - **Robusto a iluminación**: Normalización del descriptor
    
    #### Cuándo usar:
    - **Matching de objetos** con diferentes escalas/rotaciones
    - **Panoramas** y stitching de imágenes
    - **Reconocimiento de objetos** de alta precisión
    - Cuando la calidad es más importante que la velocidad
    
    ---
    
    ## 3. ORB (Oriented FAST and Rotated BRIEF)
    
    **Concepto**: Alternativa **rápida y libre de patentes** a SIFT, combina detector FAST con descriptor BRIEF mejorado.
    
    #### Características:
    - **Muy rápido** - 10-100x más rápido que SIFT
    - **Libre de patentes** - completamente gratuito
    - **Descriptor binario** - 256 bits (muy compacto)
    - **Invariante a rotación**
    - **Menos robusto a escala** que SIFT
    
    #### Componentes:
    - **FAST (Features from Accelerated Segment Test)**: Detector de esquinas rápido
    - **BRIEF (Binary Robust Independent Elementary Features)**: Descriptor binario
    - **Orientación**: Calcula usando intensidad centroid
    
    #### Parámetros importantes:
    - **nfeatures**: Número máximo de keypoints (default: 500)
    - **scaleFactor**: Factor de escala de pirámide (default: 1.2)
    - **nlevels**: Número de niveles de pirámide (default: 8)
    
    #### Cuándo usar:
    - **Aplicaciones en tiempo real** (móviles, embedded)
    - **Tracking de objetos** rápido
    - **SLAM** (Simultaneous Localization and Mapping)
    - Cuando necesitas velocidad sin sacrificar demasiada calidad
    
    ---
    
    ## 4. BRISK (Binary Robust Invariant Scalable Keypoints)
    
    **Concepto**: Detector con descriptores binarios, diseñado para ser **rápido y robusto**.
    
    #### Características:
    - **Rápido** - comparable a ORB
    - **Descriptor binario** (512 bits)
    - **Invariante a escala y rotación**
    - **Patrón de muestreo único**
    - **Descriptores más largos** que ORB (512 vs 256 bits)
    
    #### Diferencias con ORB:
    - Usa patrón de muestreo circular concéntrico
    - Mejor manejo de la escala
    - Descriptores ligeramente más distintivos pero más largos
    
    #### Cuándo usar:
    - Como alternativa a ORB cuando necesitas **mejor calidad**
    - **Matching** donde la precisión es importante
    - Aplicaciones que requieren **buenos descriptores binarios**
    
    ---
    
    ## 5. AKAZE (Accelerated-KAZE)
    
    **Concepto**: Versión acelerada de KAZE, usa **difusión no lineal** en lugar de Gaussiana.
    
    #### Características:
    - **Muy buen balance** velocidad/calidad
    - **Preserva bordes** (difusión no lineal)
    - **Descriptores binarios** eficientes
    - **Mejor que ORB** en muchos casos
    - **Más lento que ORB**, más rápido que SIFT
    
    #### Ventaja única:
    La difusión no lineal preserva mejor los bordes de objetos que el Gaussian blur usado por SIFT.
    
    #### Cuándo usar:
    - **Mejor alternativa moderna** a SIFT
    - Cuando necesitas **calidad cercana a SIFT** pero más rápido
    - **Imágenes con bordes importantes** (arquitectura, objetos manufacturados)
    
    ---
    
    ### Tabla Comparativa Completa
    
    | Detector | Velocidad | Calidad | Escala | Rotación | Descriptor | Patente | Uso Típico |
    |----------|-----------|---------|--------|----------|------------|---------|------------|
    | **Dense** | ⚡⚡⚡⚡⚡ | ⭐ | ❌ | ❌ | N/A | ✅ | Cobertura uniforme |
    | **SIFT** | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | ✅ | Float (128D) | ✅* | Alta precisión |
    | **ORB** | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⚠️ | ✅ | Binary (256) | ✅ | Tiempo real |
    | **BRISK** | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ | ✅ | Binary (512) | ✅ | Balance rápido |
    | **AKAZE** | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Binary (486) | ✅ | Mejor moderno |
    
    *SIFT: Patente expiró en 2020
    
    ---
    
    ### ¿Cómo Elegir el Detector Correcto?
    
    #### Por Velocidad:
    1. **Dense** (más rápido)
    2. **ORB**
    3. **BRISK**
    4. **AKAZE**
    5. **SIFT** (más lento)
    
    #### Por Calidad:
    1. **SIFT** (mejor calidad)
    2. **AKAZE**
    3. **BRISK**
    4. **ORB**
    5. **Dense** (peor calidad)
    
    #### Por Caso de Uso:
    
    | Aplicación | Detector Recomendado | Razón |
    |------------|---------------------|--------|
    | **Panoramas/Stitching** | SIFT, AKAZE | Necesita robustez a escala |
    | **Tracking en video** | ORB, BRISK | Velocidad crítica |
    | **SLAM/Robótica** | ORB | Velocidad + descriptores binarios |
    | **Reconocimiento de objetos** | SIFT, AKAZE | Calidad de matching |
    | **Realidad Aumentada** | ORB | Tiempo real en móviles |
    | **Bag of Features** | Dense + descriptor | Cobertura completa |
    | **Matching de alta precisión** | SIFT | Mejor discriminación |
    
    ---
    
    ### Código de Ejemplo Completo
    
    """)
    
    codigo_ejemplo = '''import cv2
import numpy as np

class DenseDetector():
    """Detector de grilla uniforme."""
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
    
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        
        # Crear grilla de keypoints
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                # Nota: KeyPoint usa (x, y) no (row, col)
                keypoints.append(cv2.KeyPoint(float(y), float(x), 
                                             self.initXyStep))
        
        return keypoints


class SIFTDetector():
    """Detector SIFT."""
    def __init__(self):
        # Intentar módulo xfeatures2d (OpenCV contrib)
        try:
            self.detector = cv2.xfeatures2d.SIFT_create()
        except AttributeError:
            # O usar módulo principal (OpenCV 4.4+)
            self.detector = cv2.SIFT_create()
    
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray, None)


# Cargar imagen
img = cv2.imread('image.jpg')

# 1. DENSE DETECTOR
print("Detectando con Dense...")
dense_detector = DenseDetector(step_size=20, feature_scale=20, img_bound=5)
dense_keypoints = dense_detector.detect(img)
print(f"Dense: {len(dense_keypoints)} keypoints")

# Dibujar keypoints
img_dense = cv2.drawKeypoints(
    img.copy(), 
    dense_keypoints, 
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 2. SIFT DETECTOR
print("Detectando con SIFT...")
sift_detector = SIFTDetector()
sift_keypoints = sift_detector.detect(img)
print(f"SIFT: {len(sift_keypoints)} keypoints")

img_sift = cv2.drawKeypoints(
    img.copy(), 
    sift_keypoints, 
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 3. ORB DETECTOR
print("Detectando con ORB...")
orb = cv2.ORB_create(nfeatures=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb_keypoints = orb.detect(gray, None)
print(f"ORB: {len(orb_keypoints)} keypoints")

img_orb = cv2.drawKeypoints(
    img.copy(), 
    orb_keypoints, 
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Mostrar resultados
cv2.imshow('Dense Detector', img_dense)
cv2.imshow('SIFT Detector', img_sift)
cv2.imshow('ORB Detector', img_orb)

# Analizar propiedades de keypoints SIFT
if len(sift_keypoints) > 0:
    print("\\nEstadísticas SIFT:")
    sizes = [kp.size for kp in sift_keypoints]
    responses = [kp.response for kp in sift_keypoints]
    
    print(f"Tamaño: min={min(sizes):.2f}, max={max(sizes):.2f}, "
          f"avg={np.mean(sizes):.2f}")
    print(f"Response: min={min(responses):.6f}, max={max(responses):.6f}, "
          f"avg={np.mean(responses):.6f}")
    
    # Top 10 keypoints por response
    top_keypoints = sorted(sift_keypoints, key=lambda kp: kp.response, 
                          reverse=True)[:10]
    
    print("\\nTop 10 keypoints:")
    for i, kp in enumerate(top_keypoints, 1):
        print(f"{i}. Pos: ({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), "
              f"Size: {kp.size:.2f}, Response: {kp.response:.6f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
    
    mostrar_codigo(codigo_ejemplo)
    
    st.markdown("""
    ---
    
    ### Tips y Mejores Prácticas
    
    #### 1. **Preprocesamiento**
    ```python
    # Ecualizar histograma para mejor detección
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # O usar CLAHE (mejor)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    ```
    
    #### 2. **Filtrar Keypoints de Baja Calidad**
    ```python
    # Filtrar por response mínimo
    min_response = 0.001
    keypoints = [kp for kp in keypoints if kp.response > min_response]
    
    # Tomar solo los N mejores
    n_best = 300
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:n_best]
    ```
    
    #### 3. **Matching de Keypoints**
    ```python
    # Después de detectar, extraer descriptores
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Dibujar matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], 
                                  None, flags=2)
    ```
    
    #### 4. **Optimización de Parámetros**
    ```python
    # ORB con más features para imágenes grandes
    orb = cv2.ORB_create(
        nfeatures=1000,        # Más keypoints
        scaleFactor=1.2,       # Factor de escala de pirámide
        nlevels=8,             # Niveles de pirámide
        edgeThreshold=31,      # Tamaño del borde
        firstLevel=0,          # Primer nivel de pirámide
        WTA_K=2,               # Puntos para producir descriptor
        patchSize=31           # Tamaño del patch
    )
    ```
    
    ---
    
    ### Conceptos Clave para Recordar
    
    1. **Dense vs Feature Detectors**: Dense da cobertura uniforme, detectores reales 
       encuentran puntos distintivos
    
    2. **SIFT es el gold standard**: Mejor calidad pero más lento y con patente 
       (aunque ya expiró)
    
    3. **ORB es el más práctico**: Rápido, libre, suficientemente bueno para la 
       mayoría de aplicaciones
    
    4. **Invariancias importantes**: Escala, rotación, iluminación - no todos los 
       detectores las tienen todas
    
    5. **Response indica calidad**: Keypoints con mayor response son más distintivos 
       y confiables
    
    6. **Descriptores vs Detectores**: Detector encuentra keypoints, descriptor los describe. 
       Algunos algoritmos hacen ambos (SIFT, ORB)
    
    7. **Preprocesamiento ayuda**: Ecualización de histograma, filtros de suavizado, 
       pueden mejorar la detección
    
    ---
    
    ### Referencias y Recursos
    
    - [OpenCV Feature Detection Tutorial](https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html)
    - [SIFT Paper (Lowe, 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
    - [ORB Paper (Rublee et al., 2011)](https://www.willowgarage.com/sites/default/files/orb_final.pdf)
    - [Feature Detection Comparison](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html)
    
    ### Aplicaciones Avanzadas
    
    #### Structure from Motion (SfM)
    - Reconstrucción 3D desde múltiples imágenes
    - Usa SIFT/ORB para matching entre vistas
    
    #### Visual SLAM
    - Localización y mapeo simultáneo
    - ORB-SLAM es uno de los más populares
    
    #### Image Stitching
    - Crear panoramas
    - Usa SIFT/AKAZE para alineación robusta
    
    #### Object Recognition
    - Identificar objetos en escenas
    - Bag of Features con descriptores SIFT/ORB
    
    #### Augmented Reality
    - Tracking de marcadores
    - ORB para detección en tiempo real
    """)


# ==================== FUNCIONES AUXILIARES ====================

def detectar_keypoints(img, detector_name, config, draw_rich=True, color=None):
    """
    Detecta keypoints usando el detector especificado.
    
    Returns:
        Tupla (keypoints, img_with_keypoints, tiempo)
    """
    import time
    
    # Crear detector
    start_time = time.time()
    
    if detector_name == "Dense":
        detector = DenseDetector(**config)
    elif detector_name == "SIFT":
        detector = SIFTDetector()
    elif detector_name == "ORB":
        detector = ORBDetector(**config)
    elif detector_name == "BRISK":
        detector = BRISKDetector()
    elif detector_name == "AKAZE":
        detector = AKAZEDetector()
    else:
        raise ValueError(f"Detector desconocido: {detector_name}")
    
    # Detectar
    keypoints = detector.detect(img)
    tiempo = time.time() - start_time
    
    # Dibujar keypoints
    img_result = np.copy(img)
    
    if color:
        # Convertir color hex a BGR
        color_bgr = hex_to_bgr(color)
    else:
        color_bgr = (0, 255, 0)
    
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_rich else 0
    
    img_result = cv2.drawKeypoints(
        img_result,
        keypoints,
        None,
        color=color_bgr,
        flags=flags
    )
    
    return keypoints, img_result, tiempo


def crear_heatmap_keypoints(img, positions):
    """Crea un mapa de calor de la densidad de keypoints."""
    rows, cols = img.shape[:2]
    
    # Crear mapa de densidad
    heatmap = np.zeros((rows, cols), dtype=np.float32)
    
    # Añadir gaussiana en cada keypoint
    for x, y in positions:
        x, y = int(x), int(y)
        if 0 <= x < cols and 0 <= y < rows:
            # Añadir punto
            y1 = max(0, y - 10)
            y2 = min(rows, y + 10)
            x1 = max(0, x - 10)
            x2 = min(cols, x + 10)
            heatmap[y1:y2, x1:x2] += 1
    
    # Aplicar blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Normalizar
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    
    # Aplicar colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer sobre imagen original
    alpha = 0.6
    result = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
    return result


def hex_to_bgr(hex_color):
    """Convierte color hex a BGR."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def cargar_imagen_input(key_base="img_source_features"):
    """Carga imagen desde archivo o upload."""
    with st.sidebar:
        st.markdown("### Cargar Imagen")
        
        opcion = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key=key_base
        )
        
        if opcion == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen",
                key=f"upload_{key_base}"
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
                # Crear imagen de ejemplo con características
                return crear_imagen_ejemplo()


def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo con características visuales."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Dibujar formas con esquinas y bordes
    # Rectángulos
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
    cv2.rectangle(img, (450, 50), (550, 150), (80, 80, 80), 2)
    
    # Círculos
    cv2.circle(img, (300, 100), 40, (120, 120, 120), -1)
    cv2.circle(img, (300, 100), 50, (60, 60, 60), 3)
    
    # Triángulos
    pts1 = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    cv2.fillPoly(img, [pts1], (90, 90, 90))
    
    # Estrella (muchos puntos de interés)
    center = (300, 300)
    outer_r, inner_r = 50, 20
    star_pts = []
    for i in range(10):
        angle = i * np.pi / 5 - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        star_pts.append([x, y])
    cv2.fillPoly(img, [np.array(star_pts, np.int32)], (70, 70, 70))
    
    # Líneas (bordes)
    cv2.line(img, (450, 250), (550, 350), (50, 50, 50), 3)
    cv2.line(img, (550, 250), (450, 350), (50, 50, 50), 3)
    
    # Texto (muchas esquinas)
    cv2.putText(img, "FEATURES", (180, 380),
                cv2.FONT_HERSHEY_BOLD, 1.2, (40, 40, 40), 3)
    
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