"""
Capítulo 7 - Ejercicio 7: Censura de Formas por Solidity Factor
Aprende a detectar y censurar formas específicas usando análisis de convex hull
y clustering K-means sin necesidad de templates
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


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Censura de Formas por Solidity Factor")
    st.markdown("""
    Detecta y censura formas específicas sin usar templates, utilizando análisis de 
    **convex hull** y **K-means clustering** basado en el factor de solidez.
    """)
    
    st.markdown("---")
    
    # Cargar imagen
    img = cargar_imagen_input()
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "Detector Interactivo",
        "Análisis de Formas",
        "Teoría"
    ])
    
    with tab1:
        detector_interactivo(img)
    
    with tab2:
        analisis_formas(img)
    
    with tab3:
        mostrar_teoria()


def detector_interactivo(img):
    """Modo interactivo de detección y censura."""
    
    crear_seccion("Controles de Detección", "")
    
    col_control, col_preview = st.columns([1, 2])
    
    with col_control:
        with panel_control("Parámetros de Detección"):
            
            # Umbral de binarización
            umbral_bin = control_slider(
                "Umbral de Binarización",
                0, 255, 170,
                "Umbral para convertir imagen a binaria",
                key="umbral_bin"
            )
            
            st.markdown("---")
            
            # Número de clusters K-means
            num_clusters = entrada_numero(
                "Número de Clusters (K)",
                2, 5, 2, 1,
                ayuda="Número de grupos para clasificar formas",
                key="num_clusters"
            )
            
            st.markdown("---")
            
            # Opciones de visualización
            mostrar_contornos = checkbox_simple(
                "Mostrar todos los contornos",
                True,
                key="show_all_contours"
            )
            
            mostrar_hull = checkbox_simple(
                "Mostrar convex hulls",
                False,
                key="show_hulls"
            )
            
            mostrar_solidity = checkbox_simple(
                "Mostrar valores de solidity",
                True,
                key="show_solidity"
            )
            
            st.markdown("---")
            
            # Método de censura
            metodo_censura = selector_opciones(
                "Método de Censura",
                ["Rectángulo Negro", "Rectángulo Rotado", "Relleno del Contorno", "Difuminado"],
                key="metodo_censura"
            )
    
    with col_preview:
        # Procesar imagen
        resultados = procesar_deteccion(
            img,
            umbral_bin,
            num_clusters,
            mostrar_contornos,
            mostrar_hull,
            mostrar_solidity,
            metodo_censura
        )
        
        if resultados is None:
            st.error("No se detectaron formas en la imagen")
            return
        
        img_contornos, img_censurada, solidity_values, labels, detected_contours = resultados
        
        # Mostrar resultados
        crear_seccion("Resultados", "")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Contornos Detectados**")
            mostrar_imagen_streamlit(img_contornos, "")
            st.info(f"Total de formas: {len(solidity_values)}")
        
        with col2:
            st.markdown("**Imagen Censurada**")
            mostrar_imagen_streamlit(img_censurada, "")
            st.info(f"Formas censuradas: {len(detected_contours)}")
        
        # Botones de acción
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if boton_accion("Guardar contornos", key="save_contours"):
                guardar_resultado(img_contornos, "formas_detectadas.jpg")
        
        with col_btn2:
            if boton_accion("Guardar censurada", key="save_censored"):
                guardar_resultado(img_censurada, "imagen_censurada.jpg")


def analisis_formas(img):
    """Análisis detallado de las formas detectadas."""
    
    crear_seccion("Análisis de Formas", "")
    
    with panel_control("Configuración"):
        umbral = control_slider(
            "Umbral de Binarización",
            0, 255, 127,
            key="umbral_analisis"
        )
    
    # Obtener contornos
    contours = obtener_contornos(img, umbral)
    
    if len(contours) == 0:
        st.warning("No se detectaron formas en la imagen")
        return
    
    # Calcular solidity para cada forma
    datos_formas = []
    
    for i, contour in enumerate(contours):
        area_contour = cv2.contourArea(contour)
        
        if area_contour < 100:  # Filtrar formas muy pequeñas
            continue
        
        convex_hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(convex_hull)
        
        if area_hull > 0:
            solidity = float(area_contour) / area_hull
            
            # Calcular otras propiedades
            perimetro = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            datos_formas.append({
                'id': i,
                'solidity': solidity,
                'area': area_contour,
                'area_hull': area_hull,
                'perimetro': perimetro,
                'aspect_ratio': aspect_ratio,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
    
    if len(datos_formas) == 0:
        st.warning("No se detectaron formas válidas")
        return
    
    
    # Estadísticas
    st.markdown("---")
    crear_seccion("Estadísticas de Solidity", "")
    
    col1, col2, col3, col4 = st.columns(4)
    
    solidity_values = [d['solidity'] for d in datos_formas]
    
    col1.metric("Mínimo", f"{min(solidity_values):.3f}")
    col2.metric("Máximo", f"{max(solidity_values):.3f}")
    col3.metric("Media", f"{np.mean(solidity_values):.3f}")
    col4.metric("Desv. Std", f"{np.std(solidity_values):.3f}")
    
    # Visualización con matplotlib
    st.markdown("---")
    crear_seccion("Distribución de Solidity", "")
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma
    ax1.hist(solidity_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Solidity Factor')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Solidity')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(solidity_values), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(solidity_values):.3f}')
    ax1.legend()
    
    # Scatter plot: Área vs Solidity
    areas = [d['area'] for d in datos_formas]
    scatter = ax2.scatter(areas, solidity_values, c=solidity_values, cmap='viridis', 
                s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('Área')
    ax2.set_ylabel('Solidity Factor')
    ax2.set_title('Área vs Solidity')
    ax2.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax2, label='Solidity')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Visualización de formas individuales
    st.markdown("---")
    crear_seccion("Explorador de Formas Individuales", "")
    
    forma_seleccionada = st.selectbox(
        "Selecciona una forma para analizar",
        range(len(datos_formas)),
        format_func=lambda x: f"Forma {x} (Solidity: {datos_formas[x]['solidity']:.3f})"
    )
    
    # Mostrar forma seleccionada
    dato = datos_formas[forma_seleccionada]
    contour = contours[dato['id']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualizar forma
        img_forma = np.copy(img)
        cv2.drawContours(img_forma, [contour], -1, (0, 255, 0), 3)
        
        # Convex hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(img_forma, [hull], -1, (255, 0, 0), 2)
        
        # Bounding box
        x, y, w, h = dato['x'], dato['y'], dato['w'], dato['h']
        cv2.rectangle(img_forma, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        mostrar_imagen_streamlit(img_forma, "Forma Seleccionada")
        
        st.markdown("""
        - **Verde**: Contorno original
        - **Azul**: Convex Hull
        - **Rojo**: Bounding Box
        """)
    
    with col2:
        st.markdown("### Propiedades")
        st.markdown(f"""
        - **ID**: {dato['id']}
        - **Solidity Factor**: `{dato['solidity']:.4f}`
        - **Área Contorno**: `{dato['area']:.2f}` px²
        - **Área Hull**: `{dato['area_hull']:.2f}` px²
        - **Diferencia**: `{dato['area_hull'] - dato['area']:.2f}` px²
        - **Perímetro**: `{dato['perimetro']:.2f}` px
        - **Aspect Ratio**: `{dato['aspect_ratio']:.3f}`
        - **Posición**: `({dato['x']}, {dato['y']})`
        - **Dimensiones**: `{dato['w']} × {dato['h']}` px
        """)
        
        # Interpretación del solidity
        if dato['solidity'] > 0.95:
            st.success("Forma muy sólida (casi convexa)")
        elif dato['solidity'] > 0.80:
            st.info("Forma moderadamente sólida")
        else:
            st.warning("Forma cóncava (baja solidity)")


def mostrar_teoria():
    """Explicación teórica del método."""
    
    crear_seccion("Teoría: Detección de Formas sin Templates", "")
    
    st.markdown("""
    ### El Problema
    
    Imagina que necesitas detectar y censurar formas específicas en una imagen (por ejemplo, 
    formas de boomerang), pero **no tienes ninguna imagen de referencia (template)** para 
    hacer shape matching. ¿Cómo lo resuelves?
    
    ### La Solución: Análisis de Propiedades Geométricas
    
    En lugar de comparar con un template, analizamos **propiedades intrínsecas** de las formas:
    
    1. **Convex Hull** (Envolvente Convexa)
    2. **Solidity Factor** (Factor de Solidez)
    3. **K-Means Clustering** (Agrupamiento No Supervisado)
    
    ### ¿Qué es el Convex Hull?
    
    El **convex hull** es el polígono convexo más pequeño que contiene completamente una forma.
    Piensa en él como una liga elástica estirada alrededor de la forma.
    
    ```
    Forma Original:     Convex Hull:
        ○─○                 ●━━━●
       ╱   ╲               ╱     ╲
      ○     ○─○          ●       ●
       ╲   ╱             ╲     ╱
        ○─○               ●━━━●
    ```
    
    ### Solidity Factor (Factor de Solidez)
    
    El **solidity factor** mide qué tan "sólida" o "completa" es una forma:
    
    ```
    Solidity = Área del Contorno / Área del Convex Hull
    ```
    
    **Interpretación**:
    - **Solidity ≈ 1.0**: Forma casi perfectamente convexa (círculo, rectángulo)
    - **Solidity < 0.8**: Forma cóncava con "huecos" o "hendiduras" (boomerang, estrella)
    
    #### Ejemplos:
    
    | Forma | Solidity | Razón |
    |-------|----------|-------|
    | Cuadrado | ~1.0 | Forma convexa perfecta |
    | Círculo | ~1.0 | Sin concavidades |
    | Estrella | ~0.5-0.7 | Puntas crean área vacía |
    | Boomerang | ~0.6-0.8 | Forma curva con concavidad |
    | Media Luna | ~0.5-0.6 | Gran área cóncava |
    
    ### K-Means Clustering
    
    **K-means** es un algoritmo de aprendizaje automático **no supervisado** que agrupa 
    datos similares en K clusters (grupos).
    
    #### ¿Por qué usarlo?
    
    - No necesitamos un umbral fijo de solidity
    - Se adapta automáticamente a diferentes conjuntos de formas
    - Separa las formas en grupos naturales
    
    #### Proceso:
    
    1. Calculamos solidity para todas las formas
    2. Aplicamos K-means (K=2 para "formas objetivo" vs "otras formas")
    3. Identificamos el cluster con **menor solidity promedio**
    4. Ese cluster contiene nuestras formas objetivo (ej: boomerangs)
    
    ### Pipeline Completo del Algoritmo
    
    ```
    1. Imagen Original
         ↓
    2. Preprocesamiento
       - Convertir a escala de grises
       - Aplicar umbral/binarización
         ↓
    3. Detección de Contornos
       - cv2.findContours()
         ↓
    4. Para cada contorno:
       - Calcular área del contorno
       - Calcular convex hull
       - Calcular área del hull
       - Solidity = área_contorno / área_hull
         ↓
    5. Clustering K-Means
       - Input: lista de solidity values
       - Output: etiquetas de cluster para cada forma
         ↓
    6. Identificar Cluster Objetivo
       - Cluster con menor centro (lowest solidity)
         ↓
    7. Censurar Formas
       - Dibujar rectángulos negros
       - O rellenar contornos
    ```
    
    ### Código Conceptual
    
    """)
    
    codigo = '''import cv2
import numpy as np

# 1. Leer y preprocesar imagen
img = cv2.imread('shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 2. Detectar contornos
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)

# 3. Calcular solidity para cada forma
solidity_values = []
for contour in contours:
    # Área del contorno original
    area_contour = cv2.contourArea(contour)
    
    # Calcular convex hull
    convex_hull = cv2.convexHull(contour)
    area_hull = cv2.contourArea(convex_hull)
    
    # Solidity factor
    solidity = float(area_contour) / area_hull
    solidity_values.append(solidity)

# 4. Aplicar K-Means clustering
solidity_array = np.array(solidity_values).reshape((-1, 1)).astype('float32')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

compactness, labels, centers = cv2.kmeans(
    solidity_array, 
    2,  # K=2 clusters
    None,
    criteria, 
    10, 
    flags
)

# 5. Identificar cluster con menor solidity (formas cóncavas)
closest_class = np.argmin(centers)

# 6. Obtener contornos del cluster objetivo
output_contours = []
for i, label in enumerate(labels):
    if label == closest_class:
        output_contours.append(contours[i])

# 7. Censurar las formas detectadas
for contour in output_contours:
    # Método 1: Rectángulo rotado mínimo
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0,0,0), -1)
    
    # Método 2: Rellenar el contorno directamente
    # cv2.drawContours(img, [contour], 0, (0,0,0), -1)

# Mostrar resultado
cv2.imshow('Censored', img)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo)
    
    st.markdown("""
    ### Métodos de Censura
    
    Una vez detectadas las formas, podemos censurarlas de diferentes maneras:
    
    | Método | Función OpenCV | Ventajas | Desventajas |
    |--------|---------------|----------|-------------|
    | **Rectángulo Rotado** | `cv2.minAreaRect()` + `cv2.boxPoints()` | Cubre bien la forma | Puede cubrir área extra |
    | **Relleno de Contorno** | `cv2.drawContours(..., -1)` | Cubre exactamente la forma | Puede verse irregular |
    | **Rectángulo Alineado** | `cv2.boundingRect()` | Simple y rápido | Cubre mucha área extra |
    | **Difuminado** | `cv2.GaussianBlur()` en ROI | Menos obvio | Más complejo |
    
    ### Aplicaciones Prácticas
    
    Este método es útil para:
    
    - **Censura automática** de objetos en imágenes
    - **Control de calidad** en manufactura (detectar piezas defectuosas)
    - **Videojuegos** (detectar colisiones con formas irregulares)
    - **Análisis médico** (identificar células anormales)
    - **Clasificación de productos** por forma
    - **Análisis de hojas** en botánica
    
    ### Limitaciones y Consideraciones
    
    - **No funciona con todas las formas**: Este método específico es ideal para formas 
      cóncavas vs convexas, pero otras situaciones requieren diferentes métricas
    - **Sensible al ruido**: Formas ruidosas pueden alterar el solidity
    - **Requiere buena segmentación**: La binarización debe ser limpia
    - **K debe ser conocido**: Necesitas saber cuántos grupos esperar
    
    ### Otras Métricas Útiles
    
    Dependiendo del caso, puedes usar:
    
    - **Extent**: `área_contorno / área_bounding_rect`
    - **Aspect Ratio**: `ancho / alto`
    - **Compactness**: `perímetro² / área`
    - **Hu Moments**: Descriptores invariantes a rotación/escala
    - **Circularity**: `4π × área / perímetro²`
    
    ### Tips para Mejores Resultados
    
    1. **Preprocesamiento**: Usa filtros de suavizado antes de binarizar
    2. **Filtrado de ruido**: Descarta contornos muy pequeños (área < 100 px)
    3. **Múltiples métricas**: Combina solidity con otras propiedades
    4. **Validación visual**: Siempre verifica los resultados manualmente
    5. **Ajuste de K**: Prueba diferentes valores de K si los resultados no son buenos
    """)
    
    st.markdown("---")
    crear_seccion("Referencias y Recursos", "")
    
    st.markdown("""
    ### Documentación Relevante
    
    - [OpenCV K-Means Tutorial](http://docs.opencv.org/master/de/d4d/tutorial_py_kmeans_understanding.html)
    - [Contour Features](https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html)
    - [Convex Hull](https://docs.opencv.org/master/d7/d1d/tutorial_hull.html)
    
    ### Conceptos Clave para Recordar
    
    - **Solidity** mide qué tan "llena" está una forma
    - **K-means** agrupa automáticamente formas similares
    - **No necesitas templates** para detectar formas específicas
    - **Las propiedades geométricas** son poderosas para clasificación
    """)


def procesar_deteccion(img, umbral, num_clusters, mostrar_todos, mostrar_hull, 
                       mostrar_solidity, metodo_censura):
    """
    Procesa la detección y censura de formas.
    
    Returns:
        Tupla (img_contornos, img_censurada, solidity_values, labels, detected_contours)
    """
    # Obtener contornos
    contours = obtener_contornos(img, umbral)
    
    if len(contours) == 0:
        return None
    
    # Calcular solidity
    solidity_values = []
    valid_contours = []
    hulls = []
    
    for contour in contours:
        area_contour = cv2.contourArea(contour)
        
        if area_contour < 100:  # Filtrar contornos muy pequeños
            continue
        
        convex_hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(convex_hull)
        
        if area_hull > 0:
            solidity = float(area_contour) / area_hull
            solidity_values.append(solidity)
            valid_contours.append(contour)
            hulls.append(convex_hull)
    
    if len(solidity_values) < 2:
        return None
    
    # Imagen para mostrar contornos
    img_contornos = np.copy(img)
    
    # Mostrar todos los contornos
    if mostrar_todos:
        cv2.drawContours(img_contornos, valid_contours, -1, (0, 255, 0), 2)
    
    # Mostrar convex hulls
    if mostrar_hull:
        cv2.drawContours(img_contornos, hulls, -1, (255, 0, 0), 2)
    
    # Mostrar valores de solidity
    if mostrar_solidity:
        for i, contour in enumerate(valid_contours):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                text = f"{solidity_values[i]:.2f}"
                cv2.putText(img_contornos, text, (cx-20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Aplicar K-means
    solidity_array = np.array(solidity_values).reshape((-1, 1)).astype('float32')
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    compactness, labels, centers = cv2.kmeans(
        solidity_array,
        int(num_clusters),
        None,
        criteria,
        10,
        flags
    )
    
    # Identificar cluster con menor solidity
    closest_class = np.argmin(centers)
    
    # Obtener contornos del cluster objetivo
    detected_contours = []
    for i in range(len(valid_contours)):
        if labels[i] == closest_class:
            detected_contours.append(valid_contours[i])
    
    # Resaltar contornos detectados en imagen de contornos
    cv2.drawContours(img_contornos, detected_contours, -1, (0, 0, 255), 3)
    
    # Crear imagen censurada
    img_censurada = np.copy(img)
    
    for contour in detected_contours:
        if metodo_censura == "Rectángulo Negro":
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_censurada, (x, y), (x+w, y+h), (0, 0, 0), -1)
        
        elif metodo_censura == "Rectángulo Rotado":
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_censurada, [box], 0, (0, 0, 0), -1)
        
        elif metodo_censura == "Relleno del Contorno":
            cv2.drawContours(img_censurada, [contour], 0, (0, 0, 0), -1)
        
        elif metodo_censura == "Difuminado":
            # Crear máscara para el contorno
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Aplicar blur solo en la región del contorno
            blurred = cv2.GaussianBlur(img_censurada, (51, 51), 30)
            img_censurada = np.where(mask[:,:,np.newaxis] == 255, blurred, img_censurada)
    
    return img_contornos, img_censurada, solidity_values, labels, detected_contours


def obtener_contornos(img, umbral):
    """Obtiene contornos de la imagen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    return contours


def cargar_imagen_input():
    """Carga imagen desde archivo o upload."""
    with st.sidebar:
        st.markdown("### Cargar Imagen")
        
        opcion = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key="img_source"
        )
        
        if opcion == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen con formas",
                key="upload_shapes"
            )
            if archivo:
                return cargar_imagen_desde_upload(archivo)
            else:
                return None
        else:
            img_path = Path("data/images/shapes.png")
            if img_path.exists():
                return leer_imagen(str(img_path))
            else:
                # Crear imagen de ejemplo con formas
                return crear_imagen_ejemplo()


def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo con diferentes formas."""
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    # Formas convexas (alta solidity)
    # Rectángulo
    cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
    
    # Círculo
    cv2.circle(img, (250, 100), 50, (100, 100, 100), -1)
    
    # Triángulo
    pts_triangle = np.array([[400, 50], [350, 150], [450, 150]], np.int32)
    cv2.fillPoly(img, [pts_triangle], (100, 100, 100))
    
    # Formas cóncavas (baja solidity) - tipo boomerang
    # Forma en C (boomerang 1)
    cv2.ellipse(img, (100, 300), (60, 60), 0, 45, 315, (100, 100, 100), 30)
    
    # Forma en media luna (boomerang 2)
    cv2.ellipse(img, (300, 300), (70, 70), 0, 0, 360, (100, 100, 100), -1)
    cv2.ellipse(img, (320, 300), (60, 60), 0, 0, 360, (255, 255, 255), -1)
    
    # Forma en U (boomerang 3)
    pts_u = np.array([
        [450, 250], [450, 350], [480, 350], [480, 280],
        [520, 280], [520, 350], [550, 350], [550, 250]
    ], np.int32)
    cv2.fillPoly(img, [pts_u], (100, 100, 100))
    
    # Estrella (baja solidity)
    center = (120, 420)
    outer_radius = 40
    inner_radius = 20
    num_points = 5
    
    star_points = []
    for i in range(num_points * 2):
        angle = i * np.pi / num_points - np.pi / 2
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        star_points.append([x, y])
    
    star_points = np.array(star_points, np.int32)
    cv2.fillPoly(img, [star_points], (100, 100, 100))
    
    # Añadir título
    cv2.putText(img, "Formas para Analisis", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
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