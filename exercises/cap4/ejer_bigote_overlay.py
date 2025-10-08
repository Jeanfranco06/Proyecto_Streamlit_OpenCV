"""
Capítulo 4 - Ejercicio 4: Superposición de Bigote
Aprende a detectar bocas y superponer accesorios usando Haar Cascades y máscaras
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

# Intentar importar streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Superposición de Bigote")
    st.markdown("""
    Detecta bocas en imágenes o video en tiempo real y superpón bigotes usando Haar Cascades 
    y operaciones de máscaras bitwise. ¡Diviértete agregando bigotes a cualquier rostro!
    """)
    
    st.markdown("---")
    
    # Verificar archivos necesarios
    verificar_archivos_necesarios()
    
    # Tabs principales
    if WEBRTC_AVAILABLE:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Modo Imagen",
            "Webcam en Vivo",
            "Proceso Técnico",
            "Teoría"
        ])
    else:
        tab1, tab2, tab3 = st.tabs([
            "Modo Imagen",
            "Proceso Técnico",
            "Teoría"
        ])
        st.info("Instala `streamlit-webrtc` para habilitar modo webcam: `pip install streamlit-webrtc`")
    
    with tab1:
        modo_imagen()
    
    if WEBRTC_AVAILABLE:
        with tab2:
            modo_webcam()
        with tab3:
            proceso_tecnico()
        with tab4:
            mostrar_teoria()
    else:
        with tab2:
            proceso_tecnico()
        with tab3:
            mostrar_teoria()


def verificar_archivos_necesarios():
    """Verifica que existan los archivos necesarios."""
    
    cascade_path = Path("data/cascade_files/haarcascade_mcs_mouth.xml")
    
    if not cascade_path.exists():
        st.error(f"No se encontró el archivo Haar Cascade: {cascade_path}")
        st.info("""
        **Cómo obtener el archivo:**
        
        1. Descarga desde el repositorio oficial de OpenCV:
           https://github.com/opencv/opencv/tree/master/data/haarcascades
        
        2. Busca el archivo `haarcascade_mcs_mouth.xml`
        
        3. Guárdalo en: `data/cascade_files/haarcascade_mcs_mouth.xml`
        """)
        return False
    
    return True


def modo_imagen():
    """Modo de procesamiento de imagen estática."""
    
    crear_seccion("Procesamiento de Imagen Estática", "")
    
    # Sidebar para controles
    with st.sidebar:
        st.markdown("### Configuración")
        
        # Selector de imagen
        opcion_imagen = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key="img_source_moustache"
        )
        
        if opcion_imagen == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube una foto con rostros",
                key="upload_moustache"
            )
            if archivo:
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("Por favor sube una imagen")
                return
        else:
            img_path = Path("data/images/face_sample.jpg")
            if img_path.exists():
                img = leer_imagen(str(img_path))
            else:
                st.error("No se encontró imagen de ejemplo")
                st.info("Sube tu propia imagen usando la opción 'Subir imagen'")
                return
        
        if img is None:
            return
        
        st.markdown("---")
        
        # Selector de bigote
        bigotes_disponibles = obtener_bigotes_disponibles()
        bigote_seleccionado = selector_opciones(
            "Estilo de Bigote",
            list(bigotes_disponibles.keys()),
            key="bigote_style"
        )
        
        st.markdown("---")
        st.markdown("### Ajustes de Detección")
        
        # Parámetros de detección
        scale_factor = entrada_numero(
            "Factor de Escala",
            1.1, 2.0, 1.3, 0.1,
            formato="%.1f",
            ayuda="Cuánto se reduce la imagen en cada escala",
            key="scale_factor"
        )
        
        min_neighbors = control_slider(
            "Vecinos Mínimos",
            1, 10, 5,
            "Cuántos vecinos debe tener cada candidato",
            key="min_neighbors"
        )
        
        st.markdown("---")
        st.markdown("### 📐 Ajustes de Posición")
        
        # Ajustes de tamaño
        width_scale = control_slider(
            "Ancho del Bigote (%)",
            50, 200, 120,
            key="width_scale"
        ) / 100.0
        
        height_scale = control_slider(
            "Alto del Bigote (%)",
            30, 100, 60,
            key="height_scale"
        ) / 100.0
        
        # Ajustes de posición
        x_offset = control_slider(
            "Desplazamiento Horizontal (%)",
            -20, 20, -5,
            key="x_offset"
        ) / 100.0
        
        y_offset = control_slider(
            "Desplazamiento Vertical (%)",
            -100, 0, -55,
            key="y_offset"
        ) / 100.0
        
        st.markdown("---")
        
        # Opciones de visualización
        mostrar_rectangulos = checkbox_simple(
            "Mostrar rectángulos de detección",
            False,
            key="show_rects"
        )
        
        detectar_todas = checkbox_simple(
            "Detectar todas las bocas",
            True,
            "Si está desactivado, solo procesará la primera boca detectada",
            key="detect_all"
        )
    
    # Procesar imagen
    moustache_path = bigotes_disponibles[bigote_seleccionado]
    
    if not moustache_path.exists():
        st.error(f"No se encontró la imagen del bigote: {moustache_path}")
        return
    
    # Aplicar bigote
    img_result, num_detections = aplicar_bigote_imagen(
        img.copy(),
        str(moustache_path),
        scale_factor,
        min_neighbors,
        width_scale,
        height_scale,
        x_offset,
        y_offset,
        mostrar_rectangulos,
        detectar_todas
    )
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original**")
        mostrar_imagen_streamlit(img, "")
    
    with col2:
        st.markdown(f"**Con Bigote ({num_detections} detectadas)**")
        mostrar_imagen_streamlit(img_result, "")
    
    # Información
    if num_detections == 0:
        st.warning("No se detectaron bocas. Intenta ajustar los parámetros de detección.")
        info_tooltip("Reduce 'Vecinos Mínimos' o ajusta 'Factor de Escala' para mejorar la detección.")
    elif num_detections > 0:
        st.success(f"Se detectaron {num_detections} boca(s) y se aplicaron bigotes exitosamente!")
    
    # Botón de descarga
    if num_detections > 0 and boton_accion("Guardar resultado", key="save_moustache"):
        guardar_resultado(img_result, f"moustache_{bigote_seleccionado.lower().replace(' ', '_')}.jpg")


def modo_webcam():
    """Modo de procesamiento en tiempo real con webcam."""
    
    crear_seccion("Webcam en Tiempo Real", "")
    
    st.markdown("""
    Activa tu webcam y el bigote se aplicará automáticamente en tiempo real cuando detecte tu boca.
    """)
    
    # Controles en sidebar
    with st.sidebar:
        st.markdown("### Configuración Webcam")
        
        bigotes_disponibles = obtener_bigotes_disponibles()
        bigote_webcam = selector_opciones(
            "Estilo de Bigote",
            list(bigotes_disponibles.keys()),
            key="bigote_webcam"
        )
        
        scale_webcam = entrada_numero(
            "Factor de Escala",
            1.1, 2.0, 1.3, 0.1,
            key="scale_webcam"
        )
        
        neighbors_webcam = control_slider(
            "Vecinos Mínimos",
            1, 10, 5,
            key="neighbors_webcam"
        )
    
    # Configurar transformer
    moustache_path = bigotes_disponibles[bigote_webcam]
    
    class MoustacheTransformer(VideoTransformerBase):
        def __init__(self):
            self.mouth_cascade = cv2.CascadeClassifier(
                'data/cascade_files/haarcascade_mcs_mouth.xml'
            )
            self.moustache_img = cv2.imread(str(moustache_path))
            self.scale_factor = scale_webcam
            self.min_neighbors = neighbors_webcam
        
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Aplicar bigote
            result, _ = aplicar_bigote_frame(
                img,
                self.moustache_img,
                self.mouth_cascade,
                self.scale_factor,
                self.min_neighbors
            )
            
            return result
    
    # Configuración RTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Streamer
    webrtc_streamer(
        key="moustache_webcam",
        video_transformer_factory=MoustacheTransformer,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    st.info("💡 **Tip:** Muévete lentamente y asegúrate de tener buena iluminación para mejor detección.")


def proceso_tecnico():
    """Visualización paso a paso del proceso técnico."""
    
    crear_seccion("Proceso Técnico Paso a Paso", "")
    
    st.markdown("""
    Veamos en detalle cómo funciona el proceso de superposición de bigote usando máscaras bitwise.
    """)
    
    # Cargar imagen de ejemplo
    img_path = Path("data/images/face_sample.jpg")
    if img_path.exists():
        img = leer_imagen(str(img_path))
    else:
        st.warning("Necesitas una imagen de ejemplo para ver el proceso técnico")
        return
    
    # Cargar bigote
    bigotes = obtener_bigotes_disponibles()
    moustache_path = list(bigotes.values())[0]
    
    if not moustache_path.exists():
        st.error("No se encontró imagen de bigote")
        return
    
    moustache = leer_imagen(str(moustache_path))
    
    # Paso 1: Detección de boca
    st.markdown("### Paso 1: Detectar Boca con Haar Cascade")
    
    cascade_path = "data/cascade_files/haarcascade_mcs_mouth.xml"
    mouth_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(mouth_rects) == 0:
        st.warning("No se detectó ninguna boca en esta imagen")
        return
    
    # Dibujar rectángulo de detección
    img_detection = img.copy()
    for (x, y, w, h) in mouth_rects:
        cv2.rectangle(img_detection, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_detection, "BOCA", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown(f"**Boca Detectada ({len(mouth_rects)} encontrada(s))**")
        mostrar_imagen_streamlit(img_detection, "")
    
    st.info(f"Coordenadas: x={mouth_rects[0][0]}, y={mouth_rects[0][1]}, "
            f"w={mouth_rects[0][2]}, h={mouth_rects[0][3]}")
    
    st.markdown("---")
    
    # Paso 2: Ajustar dimensiones
    st.markdown("### Paso 2: Ajustar Dimensiones y Posición")
    
    (x, y, w, h) = mouth_rects[0]
    h_new = int(0.6 * h)
    w_new = int(1.2 * w)
    x_new = x - int(0.05 * w)
    y_new = y - int(0.55 * h)
    
    img_adjusted = img.copy()
    cv2.rectangle(img_adjusted, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(img_adjusted, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 2)
    cv2.putText(img_adjusted, "Original", (x, y-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img_adjusted, "Ajustada", (x_new, y_new-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    mostrar_imagen_streamlit(img_adjusted, "Verde=Detección Original, Azul=Región Ajustada")
    
    st.code(f"""
# Ajustes aplicados:
altura_nueva = 60% de altura_original = {h_new}px
ancho_nuevo = 120% de ancho_original = {w_new}px
offset_x = -5% del ancho = {x_new - x}px
offset_y = -55% de la altura = {y_new - y}px
    """)
    
    st.markdown("---")
    
    # Paso 3: Redimensionar bigote
    st.markdown("### Paso 3: Redimensionar Bigote")
    
    moustache_resized = cv2.resize(moustache, (w_new, h_new), 
                                   interpolation=cv2.INTER_AREA)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Bigote Original ({moustache.shape[1]}x{moustache.shape[0]})**")
        mostrar_imagen_streamlit(moustache, "")
    with col2:
        st.markdown(f"**Bigote Redimensionado ({w_new}x{h_new})**")
        mostrar_imagen_streamlit(moustache_resized, "")
    
    st.markdown("---")
    
    # Paso 4: Crear máscara
    st.markdown("### Paso 4: Crear Máscara Binaria")
    
    gray_moustache = cv2.cvtColor(moustache_resized, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_moustache, 50, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Bigote en Gris**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(gray_moustache, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col2:
        st.markdown("**Máscara del Bigote**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    with col3:
        st.markdown("**Máscara Invertida**")
        mostrar_imagen_streamlit(
            cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR),
            "",
            convertir_rgb=False
        )
    
    st.markdown("---")
    
    # Paso 5: Operaciones bitwise
    st.markdown("### Paso 5: Aplicar Operaciones Bitwise")
    
    # Extraer ROI
    frame_roi = img[y_new:y_new+h_new, x_new:x_new+w_new]
    
    # Operaciones bitwise
    masked_moustache = cv2.bitwise_and(moustache_resized, moustache_resized, mask=mask)
    masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ROI Original**")
        mostrar_imagen_streamlit(frame_roi, "")
    with col2:
        st.markdown("**ROI con Máscara Invertida**")
        mostrar_imagen_streamlit(masked_frame, "")
    
    st.markdown("**Bigote Extraído (con máscara)**")
    mostrar_imagen_streamlit(masked_moustache, "")
    
    st.markdown("---")
    
    # Paso 6: Combinación final
    st.markdown("### Paso 6: Combinar Imágenes")
    
    combined = cv2.add(masked_moustache, masked_frame)
    
    st.markdown("**Resultado de cv2.add(bigote_enmascarado, frame_enmascarado)**")
    mostrar_imagen_streamlit(combined, "")
    
    st.markdown("---")
    
    # Paso 7: Resultado final
    st.markdown("### Paso 7: Insertar en Frame Original")
    
    img_final = img.copy()
    img_final[y_new:y_new+h_new, x_new:x_new+w_new] = combined
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown("**Resultado Final**")
        mostrar_imagen_streamlit(img_final, "")
    
    st.success("¡Proceso completado! El bigote se ha superpuesto exitosamente.")


def mostrar_teoria():
    """Sección teórica sobre detección y superposición."""
    
    crear_seccion("Teoría: Detección y Superposición de Objetos", "")
    
    st.markdown("""
    ### ¿Qué son los Haar Cascades?
    
    Los **Haar Cascade Classifiers** son algoritmos de detección de objetos basados en 
    características Haar (propuestas por Viola y Jones en 2001). Son ampliamente usados 
    para detección de rostros y partes faciales en tiempo real.
    
    ### Conceptos Clave
    
    #### 1. **Características Haar**
    
    Son patrones rectangulares simples que detectan bordes, líneas y cambios de intensidad:
    
    - **Edge features**: Detectan bordes
    - **Line features**: Detectan líneas
    - **Four-rectangle features**: Detectan patrones más complejos
    
    #### 2. **Cascade (Cascada)**
    
    Un clasificador en cascada aplica múltiples etapas de clasificación:
    
    ```
    Imagen → Etapa 1 → Etapa 2 → ... → Etapa N → Objeto Detectado
    ```
    
    - **Ventaja**: Rechaza candidatos negativos rápidamente
    - **Resultado**: Procesamiento muy eficiente
    
    ### Método detectMultiScale()
    
    ```python
    mouth_rects = mouth_cascade.detectMultiScale(
        image,              # Imagen en escala de grises
        scaleFactor=1.3,    # Factor de reducción en cada escala
        minNeighbors=5,     # Vecinos mínimos para aceptar detección
        minSize=(30, 30)    # Tamaño mínimo del objeto
    )
    ```
    
    **Parámetros explicados:**
    
    - **scaleFactor**: 1.1 - 2.0
      - Más bajo (1.1): Más escalas, más preciso, más lento
      - Más alto (1.5-2.0): Menos escalas, más rápido, menos preciso
    
    - **minNeighbors**: 1 - 10
      - Más bajo: Más detecciones (incluye falsos positivos)
      - Más alto: Menos detecciones (más confiables)
    
    ### Operaciones Bitwise para Máscaras
    
    Las operaciones bit a bit son fundamentales para combinar imágenes:
    
    #### **AND Bitwise**
    ```python
    result = cv2.bitwise_and(img1, img2, mask=mask)
    ```
    - Mantiene solo los píxeles donde la máscara es blanca (255)
    - Usado para "extraer" regiones de interés
    
    #### **NOT Bitwise**
    ```python
    result = cv2.bitwise_not(mask)
    ```
    - Invierte la máscara (blanco→negro, negro→blanco)
    - Útil para invertir selecciones
    
    #### **ADD**
    ```python
    result = cv2.add(img1, img2)
    ```
    - Suma píxeles de dos imágenes
    - Usado para combinar elementos enmascarados
    
    ### Pipeline de Superposición
    
    ```
    1. Detectar boca → [x, y, w, h]
    2. Ajustar dimensiones → [x', y', w', h']
    3. Extraer ROI del frame → frame_roi
    4. Redimensionar bigote → bigote_pequeño
    5. Crear máscara binaria → mask y mask_inv
    6. Extraer bigote → bigote AND mask
    7. Limpiar ROI → frame_roi AND mask_inv
    8. Combinar → ADD(bigote_enmascarado, roi_limpio)
    9. Insertar en frame original
    ```
    
    ### Ajustes de Posición
    
    Los ajustes son críticos para una superposición natural:
    
    ```python
    # Dimensiones
    altura = 60% de altura_boca    # Bigote más pequeño que la boca
    ancho = 120% de ancho_boca      # Bigote más ancho que la boca
    
    # Posición
    x -= 5% del ancho               # Centrar mejor
    y -= 55% de la altura           # Mover hacia arriba (sobre la boca)
    ```
    
    ### Tips para Mejores Resultados
    
    **Iluminación uniforme** - Mejora la detección
    **Rostro frontal** - Los cascades funcionan mejor de frente
    **Imagen de bigote con fondo oscuro** - Facilita la creación de máscaras
    **Formato PNG con transparencia** - Ideal para máscaras complejas
    **Ajustar threshold** según el fondo del bigote
    
    **Limitaciones:**
    - No funciona bien con ángulos extremos
    - Requiere buena iluminación
    - Puede tener falsos positivos
    - Sensible a oclusiones parciales
    
    ### Aplicaciones Reales
    
    - **Filtros de redes sociales** (Snapchat, Instagram)
    - **Videojuegos** con avatares personalizados
    - **Prueba virtual** de accesorios (gafas, sombreros, joyas)
    - **Efectos de video** en tiempo real
    - **Cabinas de fotos** automatizadas
    - **Post-producción** de video
    
    ### Comparación con Técnicas Modernas
    
    | Técnica | Velocidad | Precisión | Complejidad |
    |---------|-----------|-----------|-------------|
    | **Haar Cascades** | ⚡⚡⚡ Muy rápido | ⭐⭐ Buena | 🟢 Baja |
    | HOG + SVM | ⚡⚡ Rápido | ⭐⭐⭐ Muy buena | 🟡 Media |
    | Deep Learning (CNN) | ⚡ Moderado | ⭐⭐⭐⭐ Excelente | 🔴 Alta |
    | MediaPipe | ⚡⚡⚡ Muy rápido | ⭐⭐⭐⭐ Excelente | 🟡 Media |
    
    **Haar Cascades sigue siendo popular por:**
    - No requiere GPU
    - Muy ligero (archivos XML pequeños)
    - Funciona en tiempo real en hardware modesto
    - Fácil de implementar
    """)
    
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo = '''import cv2
import numpy as np

# Cargar clasificador y bigote
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
moustache = cv2.imread('moustache.png')

# Captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar bocas
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(mouth_rects) > 0:
        (x, y, w, h) = mouth_rects[0]
        
        # Ajustar dimensiones
        h, w = int(0.6*h), int(1.2*w)
        x -= int(0.05*w)
        y -= int(0.55*h)
        
        # Extraer ROI y redimensionar bigote
        frame_roi = frame[y:y+h, x:x+w]
        moustache_small = cv2.resize(moustache, (w, h))
        
        # Crear máscaras
        gray_moustache = cv2.cvtColor(moustache_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_moustache, 50, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        
        # Operaciones bitwise
        masked_moustache = cv2.bitwise_and(moustache_small, moustache_small, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        
        python        # Combinar
        final = cv2.add(masked_moustache, masked_frame)
        frame[y:y+h, x:x+w] = final
    
    cv2.imshow('Moustache Filter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
    
    mostrar_codigo(codigo, "python")
    
    st.markdown("---")
    
    # Recursos adicionales
    crear_seccion("Recursos Adicionales", "")
    
    st.markdown("""
    ### Lecturas Recomendadas
    
    - [Viola-Jones Object Detection Framework (Paper Original)](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
    - [OpenCV: Cascade Classifier](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
    - [Haar-like Features (Wikipedia)](https://en.wikipedia.org/wiki/Haar-like_feature)
    
    ### Ejercicios Propuestos
    
    1. **Múltiples accesorios**: Agrega detección de ojos para colocar gafas
    2. **Diferentes estilos**: Crea tu propia colección de bigotes
    3. **Sombreros**: Detecta rostros completos y coloca sombreros
    4. **Efectos animados**: Haz que el bigote cambie de color o tamaño
    5. **Filtro de edad**: Combina con otros efectos (arrugas, canas)
    
    ### Ideas Creativas
    
    - Sistema de cabina de fotos automática
    - App de prueba de looks (bigotes, barbas, gafas)
    - Juego de "adivina quién" con filtros
    - Generador de memes automático
    - Sistema de avatares personalizados
    """)


def obtener_bigotes_disponibles():
    """Retorna diccionario de bigotes disponibles."""
    
    bigotes_dir = Path("data/accessories/moustaches")
    bigotes_dir.mkdir(parents=True, exist_ok=True)
    
    # Bigotes predefinidos
    bigotes = {
        "Clásico": bigotes_dir / "moustache_classic.png",
        "Handlebar": bigotes_dir / "moustache_handlebar.png",
        "Chevron": bigotes_dir / "moustache_chevron.png",
        "Dali": bigotes_dir / "moustache_dali.png",
        "Walrus": bigotes_dir / "moustache_walrus.png"
    }
    
    # Verificar cuáles existen
    bigotes_existentes = {}
    for nombre, path in bigotes.items():
        if path.exists():
            bigotes_existentes[nombre] = path
    
    # Si no hay ninguno, crear uno simple
    if not bigotes_existentes:
        st.warning("⚠️ No se encontraron imágenes de bigotes. Creando bigote de ejemplo...")
        bigote_default = crear_bigote_ejemplo(bigotes_dir / "moustache_classic.png")
        if bigote_default:
            bigotes_existentes["Clásico"] = bigote_default
    
    return bigotes_existentes


def crear_bigote_ejemplo(output_path):
    """Crea una imagen de bigote de ejemplo."""
    
    try:
        # Crear imagen con fondo negro
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Dibujar bigote simple (dos elipses)
        center1 = (150, 100)
        center2 = (250, 100)
        axes = (80, 60)
        
        # Bigote izquierdo
        cv2.ellipse(img, center1, axes, 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, center1, axes, 0, 0, 360, (200, 200, 200), 2)
        
        # Bigote derecho
        cv2.ellipse(img, center2, axes, 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, center2, axes, 0, 0, 360, (200, 200, 200), 2)
        
        # Centro del bigote
        cv2.rectangle(img, (180, 70), (220, 130), (255, 255, 255), -1)
        
        # Guardar
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        
        st.success(f"Bigote de ejemplo creado en: {output_path}")
        return output_path
        
    except Exception as e:
        st.error(f"Error creando bigote de ejemplo: {e}")
        return None


def aplicar_bigote_imagen(img, moustache_path, scale_factor, min_neighbors,
                          width_scale, height_scale, x_offset, y_offset,
                          mostrar_rectangulos, detectar_todas):
    """Aplica bigote a una imagen estática."""
    
    # Cargar clasificador y bigote
    cascade_path = "data/cascade_files/haarcascade_mcs_mouth.xml"
    mouth_cascade = cv2.CascadeClassifier(cascade_path)
    moustache = cv2.imread(moustache_path)
    
    if moustache is None:
        st.error("No se pudo cargar la imagen del bigote")
        return img, 0
    
    # Detectar bocas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )
    
    num_detections = len(mouth_rects)
    
    if num_detections == 0:
        return img, 0
    
    # Procesar detecciones
    mouths_to_process = mouth_rects if detectar_todas else [mouth_rects[0]]
    
    for (x, y, w, h) in mouths_to_process:
        # Dibujar rectángulo si está activado
        if mostrar_rectangulos:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calcular nuevas dimensiones y posición
        h_new = int(height_scale * h)
        w_new = int(width_scale * w)
        x_new = x + int(x_offset * w)
        y_new = y + int(y_offset * h)
        
        # Verificar límites
        if x_new < 0 or y_new < 0 or x_new + w_new > img.shape[1] or y_new + h_new > img.shape[0]:
            continue
        
        # Aplicar bigote
        img = aplicar_bigote_a_roi(img, moustache, x_new, y_new, w_new, h_new)
    
    return img, num_detections


def aplicar_bigote_a_roi(img, moustache, x, y, w, h):
    """Aplica el bigote a una región específica de la imagen."""
    
    try:
        # Extraer ROI
        frame_roi = img[y:y+h, x:x+w]
        
        # Redimensionar bigote
        moustache_resized = cv2.resize(moustache, (w, h), interpolation=cv2.INTER_AREA)
        
        # Crear máscara
        gray_moustache = cv2.cvtColor(moustache_resized, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_moustache, 50, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        
        # Operaciones bitwise
        masked_moustache = cv2.bitwise_and(moustache_resized, moustache_resized, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        
        # Combinar
        final = cv2.add(masked_moustache, masked_frame)
        
        # Insertar en imagen
        img[y:y+h, x:x+w] = final
        
    except Exception as e:
        st.error(f"Error aplicando bigote: {e}")
    
    return img


def aplicar_bigote_frame(frame, moustache, mouth_cascade, scale_factor, min_neighbors):
    """Aplica bigote a un frame de video (para webcam)."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    
    num_detections = len(mouth_rects)
    
    for (x, y, w, h) in mouth_rects:
        # Ajustes predeterminados para webcam
        h_new = int(0.6 * h)
        w_new = int(1.2 * w)
        x_new = x - int(0.05 * w)
        y_new = y - int(0.55 * h)
        
        # Verificar límites
        if x_new < 0 or y_new < 0 or x_new + w_new > frame.shape[1] or y_new + h_new > frame.shape[0]:
            continue
        
        frame = aplicar_bigote_a_roi(frame, moustache, x_new, y_new, w_new, h_new)
    
    return frame, num_detections


def guardar_resultado(img, filename):
    """Guarda el resultado en disco."""
    
    output_dir = Path("output/moustache_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    try:
        cv2.imwrite(str(output_path), img)
        st.success(f"Imagen guardada en: {output_path}")
        
        # Ofrecer descarga
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Descargar imagen",
                data=file,
                file_name=filename,
                mime="image/jpeg"
            )
    except Exception as e:
        st.error(f"Error guardando imagen: {e}")


if __name__ == "__main__":
    run()