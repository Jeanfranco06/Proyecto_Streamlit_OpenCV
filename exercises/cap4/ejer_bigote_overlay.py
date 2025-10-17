"""
Capítulo 4 - Ejercicio 4: Superposición de Bigote (VERSIÓN ROBUSTA)
Detecta bocas y superpone accesorios usando Haar Cascades y máscaras
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import urllib.request
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


# URLs de los cascades de OpenCV
CASCADES_URLS = {
    'mouth': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_mcs_mouth.xml',
    'face': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
}


def descargar_cascade(cascade_name):
    """Descarga un cascade de OpenCV si no existe localmente."""
    cascade_dir = Path("data/cascade_files")
    cascade_dir.mkdir(parents=True, exist_ok=True)
    
    if cascade_name == 'mouth':
        cascade_file = cascade_dir / "haarcascade_mcs_mouth.xml"
        url = CASCADES_URLS['mouth']
    elif cascade_name == 'face':
        cascade_file = cascade_dir / "haarcascade_frontalface_default.xml"
        url = CASCADES_URLS['face']
    else:
        return None
    
    # Si ya existe, retornar
    if cascade_file.exists():
        return str(cascade_file)
    
    try:
        st.info(f"Descargando {cascade_name} cascade...")
        urllib.request.urlretrieve(url, cascade_file)
        st.success(f"✅ {cascade_name} cascade descargado correctamente")
        return str(cascade_file)
    except Exception as e:
        st.error(f"❌ Error descargando cascade: {e}")
        return None


def obtener_cascade(cascade_name):
    """Obtiene la ruta del cascade, descargándolo si es necesario."""
    cascade_dir = Path("data/cascade_files")
    
    if cascade_name == 'mouth':
        cascade_file = cascade_dir / "haarcascade_mcs_mouth.xml"
    elif cascade_name == 'face':
        cascade_file = cascade_dir / "haarcascade_frontalface_default.xml"
    else:
        return None
    
    if cascade_file.exists():
        return str(cascade_file)
    else:
        return descargar_cascade(cascade_name)


def verificar_cascades():
    """Verifica que existan los cascades necesarios."""
    mouth_cascade = obtener_cascade('mouth')
    face_cascade = obtener_cascade('face')
    
    if not mouth_cascade or not face_cascade:
        st.error("No se pudieron obtener los archivos Haar Cascade")
        return False
    
    # Verificar que los cascades se carguen correctamente
    try:
        mouth_c = cv2.CascadeClassifier(mouth_cascade)
        face_c = cv2.CascadeClassifier(face_cascade)
        
        if mouth_c.empty() or face_c.empty():
            st.error("Los cascades descargados están vacíos o dañados")
            return False
        
        return True
    except Exception as e:
        st.error(f"Error verificando cascades: {e}")
        return False


def run():
    """Función principal del ejercicio."""
    
    st.title("Superposición de Bigote")
    st.markdown("""
    Detecta bocas en imágenes o video en tiempo real y superpón bigotes usando Haar Cascades 
    y operaciones de máscaras bitwise. ¡Diviértete agregando bigotes a cualquier rostro!
    """)
    
    st.markdown("---")
    
    # Verificar cascades
    if not verificar_cascades():
        st.error("⚠️ No se pudieron cargar los archivos necesarios")
        return
    
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
        st.info("💡 Instala `streamlit-webrtc` para habilitar modo webcam: `pip install streamlit-webrtc`")
    
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


def modo_imagen():
    """Modo de procesamiento de imagen estática."""
    
    crear_seccion("Procesamiento de Imagen Estática", "")
    
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
        if not bigotes_disponibles:
            st.error("No hay bigotes disponibles")
            return
            
        bigote_seleccionado = selector_opciones(
            "Estilo de Bigote",
            list(bigotes_disponibles.keys()),
            key="bigote_style"
        )
        
        st.markdown("---")
        st.markdown("### Ajustes de Detección")
        
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
    img_result, num_detections = aplicar_bigote_imagen_mejorado(
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
        st.warning("⚠️ No se detectaron bocas. Intenta ajustar los parámetros de detección.")
        info_tooltip("Reduce 'Vecinos Mínimos' o ajusta 'Factor de Escala' para mejorar la detección.")
    else:
        st.success(f"✅ Se detectaron {num_detections} boca(s) y se aplicaron bigotes exitosamente!")
    
    # Botón de descarga
    if num_detections > 0 and boton_accion("Guardar resultado", key="save_moustache"):
        guardar_resultado(img_result, f"moustache_{bigote_seleccionado.lower().replace(' ', '_')}.jpg")


def modo_webcam():
    """Modo de procesamiento en tiempo real con webcam."""
    
    crear_seccion("Webcam en Tiempo Real", "")
    
    st.markdown("""
    Activa tu webcam y el bigote se aplicará automáticamente en tiempo real cuando detecte tu boca.
    """)
    
    with st.sidebar:
        st.markdown("### Configuración Webcam")
        
        bigotes_disponibles = obtener_bigotes_disponibles()
        if not bigotes_disponibles:
            st.error("No hay bigotes disponibles")
            return
            
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
    
    moustache_path = bigotes_disponibles[bigote_webcam]
    
    class MoustacheTransformer(VideoTransformerBase):
        def __init__(self):
            self.mouth_cascade_path = obtener_cascade('mouth')
            self.mouth_cascade = cv2.CascadeClassifier(self.mouth_cascade_path)
            self.moustache_img = cv2.imread(str(moustache_path), cv2.IMREAD_UNCHANGED)
            self.scale_factor = scale_webcam
            self.min_neighbors = neighbors_webcam
        
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            if self.mouth_cascade.empty():
                return img
            
            result, _ = aplicar_bigote_frame(
                img,
                self.moustache_img,
                self.mouth_cascade,
                self.scale_factor,
                self.min_neighbors
            )
            
            return result
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
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
    
    img_path = Path("data/images/face_sample.jpg")
    if img_path.exists():
        img = leer_imagen(str(img_path))
    else:
        st.warning("Necesitas una imagen de ejemplo para ver el proceso técnico")
        return
    
    bigotes = obtener_bigotes_disponibles()
    if not bigotes:
        st.error("No hay bigotes disponibles")
        return
        
    moustache_path = list(bigotes.values())[0]
    
    if not moustache_path.exists():
        st.error("No se encontró imagen de bigote")
        return
    
    moustache = leer_imagen(str(moustache_path))
    
    # Usar la misma función que funciona en modo_imagen
    st.markdown("### Paso 1: Detectar Boca con Haar Cascade")
    
    cascade_path_face = obtener_cascade('face')
    cascade_path_mouth = obtener_cascade('mouth')
    
    if not cascade_path_face or not cascade_path_mouth:
        st.error("No se pudieron obtener los cascades")
        return
        
    face_cascade = cv2.CascadeClassifier(cascade_path_face)
    mouth_cascade = cv2.CascadeClassifier(cascade_path_mouth)
    
    if face_cascade.empty() or mouth_cascade.empty():
        st.error("Los cascades están vacíos")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) == 0:
        st.warning("No se detectó ningún rostro")
        return
    
    # Detectar boca dentro de la región de cara
    detected = 0
    for (fx, fy, fw, fh) in faces:
        search_x = fx
        search_y = fy + int(fh * 0.45)
        search_w = fw
        search_h = int(fh * 0.5)
        
        search_x = max(0, search_x)
        search_y = max(0, search_y)
        search_w = min(img.shape[1] - search_x, search_w)
        search_h = min(img.shape[0] - search_y, search_h)
        
        roi_gray = cv2.cvtColor(img[search_y:search_y+search_h, search_x:search_x+search_w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.equalizeHist(roi_gray)
        
        min_w = max(15, int(fw * 0.15))
        min_h = max(10, int(fh * 0.08))
        
        mouth_rects = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(min_w, min_h)
        )
        
        if len(mouth_rects) == 0:
            mouth_rects = mouth_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(min_w, min_h)
            )
        
        if len(mouth_rects) > 0:
            mx_local, my_local, mw, mh = mouth_rects[0]
            mx = search_x + mx_local
            my = search_y + my_local
            detected += 1
            break
    
    if detected == 0:
        st.warning("No se detectó ninguna boca")
        return
    
    # Visualización Paso 1
    img_detection = img.copy()
    cv2.rectangle(img_detection, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
    cv2.putText(img_detection, "BOCA", (mx, my-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown("**Boca Detectada**")
        mostrar_imagen_streamlit(img_detection, "")
    
    st.markdown("---")
    
    # Paso 2: Ajustar dimensiones
    st.markdown("### Paso 2: Ajustar Dimensiones y Posición")
    
    h_new = int(0.6 * mh)
    w_new = int(1.2 * mw)
    x_new = mx - int(0.05 * mw)
    y_new = my - int(0.55 * mh)
    
    x_new = max(0, x_new)
    y_new = max(0, y_new)
    w_new = min(w_new, img.shape[1] - x_new)
    h_new = min(h_new, img.shape[0] - y_new)
    
    img_adjusted = img.copy()
    cv2.rectangle(img_adjusted, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
    cv2.rectangle(img_adjusted, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 2)
    cv2.putText(img_adjusted, "Original", (mx, my-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img_adjusted, "Ajustada", (x_new, y_new-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    mostrar_imagen_streamlit(img_adjusted, "Verde=Boca Original, Azul=Región Ajustada para Bigote")
    
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
        gray_bgr = cv2.cvtColor(gray_moustache, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(gray_bgr, "")
    
    with col2:
        st.markdown("**Máscara del Bigote**")
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(mask_bgr, "")
    
    with col3:
        st.markdown("**Máscara Invertida**")
        mask_inv_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(mask_inv_bgr, "")
    
    st.markdown("---")
    
    # Paso 5: Operaciones bitwise
    st.markdown("### Paso 5: Aplicar Operaciones Bitwise")
    
    frame_roi = img[y_new:y_new+h_new, x_new:x_new+w_new].copy()
    
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
    
    st.success("Proceso completado! El bigote se ha superpuesto exitosamente.")


def mostrar_teoria():
    """Sección teórica sobre detección y superposición."""
    
    crear_seccion("Teoría: Detección y Superposición de Objetos", "")
    
    st.markdown("""
    ### ¿Qué son los Haar Cascades?
    
    Los **Haar Cascade Classifiers** son algoritmos de detección de objetos basados en 
    características Haar (propuestas por Viola y Jones en 2001).
    
    ### Conceptos Clave
    
    **Características Haar**: Patrones rectangulares que detectan bordes, líneas y cambios
    
    **Cascade (Cascada)**: Aplica múltiples etapas de clasificación para rechazar 
    rápidamente candidatos negativos
    
    ### Método detectMultiScale()
    
    ```python
    mouth_rects = mouth_cascade.detectMultiScale(
        image,              # Imagen en escala de grises
        scaleFactor=1.3,    # Factor de reducción en cada escala
        minNeighbors=5,     # Vecinos mínimos para aceptar detección
        minSize=(30, 30)    # Tamaño mínimo del objeto
    )
    ```
    
    ### Operaciones Bitwise
    
    - **AND**: Mantiene píxeles donde la máscara es blanca
    - **NOT**: Invierte la máscara
    - **ADD**: Suma píxeles de dos imágenes
    
    ### Pipeline de Superposición
    
    1. Detectar boca → [x, y, w, h]
    2. Ajustar dimensiones y posición
    3. Extraer región de interés (ROI)
    4. Redimensionar bigote
    5. Crear máscara binaria
    6. Aplicar operaciones bitwise
    7. Combinar y insertar en la imagen
    """)


def obtener_bigotes_disponibles():
    """Retorna diccionario de bigotes disponibles."""
    
    bigotes_dir = Path("data/accessories/moustaches")
    bigotes_dir.mkdir(parents=True, exist_ok=True)
    
    bigotes = {
        "Clásico": bigotes_dir / "moustache_classic.png",
        "Handlebar": bigotes_dir / "moustache_handlebar.png",
        "Chevron": bigotes_dir / "moustache_chevron.png",
        "Dali": bigotes_dir / "moustache_dali.png",
        "Walrus": bigotes_dir / "moustache_walrus.png"
    }
    
    bigotes_existentes = {}
    for nombre, path in bigotes.items():
        if path.exists():
            bigotes_existentes[nombre] = path
    
    if not bigotes_existentes:
        st.warning("⚠️ No se encontraron imágenes de bigotes. Creando bigote de ejemplo...")
        bigote_default = crear_bigote_ejemplo(bigotes_dir / "moustache_classic.png")
        if bigote_default:
            bigotes_existentes["Clásico"] = bigote_default
    
    return bigotes_existentes


def crear_bigote_ejemplo(output_path):
    """Crea una imagen de bigote de ejemplo."""
    
    try:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        center1 = (150, 100)
        center2 = (250, 100)
        axes = (80, 60)
        
        cv2.ellipse(img, center1, axes, 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, center1, axes, 0, 0, 360, (200, 200, 200), 2)
        cv2.ellipse(img, center2, axes, 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, center2, axes, 0, 0, 360, (200, 200, 200), 2)
        cv2.rectangle(img, (180, 70), (220, 130), (255, 255, 255), -1)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        
        st.success(f"✅ Bigote de ejemplo creado")
        return output_path
        
    except Exception as e:
        st.error(f"Error creando bigote de ejemplo: {e}")
        return None


def aplicar_bigote_imagen_mejorado(img, moustache_path, scale_factor, min_neighbors,
                                   width_scale, height_scale, x_offset, y_offset,
                                   mostrar_rectangulos, detectar_todas):
    """Aplica bigote a una imagen estática con manejo robusto de cascades."""
    
    mouth_cascade_path = obtener_cascade('mouth')
    face_cascade_path = obtener_cascade('face')
    
    if not mouth_cascade_path or not face_cascade_path:
        st.error("No se pudieron obtener los cascades")
        return img, 0
    
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if mouth_cascade.empty() or face_cascade.empty():
        st.error("Los cascades no se cargaron correctamente")
        return img, 0
    
    moustache = cv2.imread(moustache_path, cv2.IMREAD_UNCHANGED)
    
    if moustache is None:
        st.error("No se pudo cargar la imagen del bigote")
        return img, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    detected = 0
    
    if len(faces) == 0:
        mouth_rects = mouth_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        mouths_to_process = mouth_rects if detectar_todas else ([] if len(mouth_rects)==0 else [mouth_rects[0]])
        
        for (x, y, w, h) in mouths_to_process:
            if mostrar_rectangulos:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            h_new = int(height_scale * h)
            w_new = int(width_scale * w)
            x_new = x + int(x_offset * w)
            y_new = y + int(y_offset * h)
            if x_new < 0 or y_new < 0 or x_new + w_new > img.shape[1] or y_new + h_new > img.shape[0]:
                continue
            img = aplicar_bigote_a_roi_mejorado(img, moustache, x_new, y_new, w_new, h_new)
            detected += 1
        return img, detected
    
    for (fx, fy, fw, fh) in faces:
        search_x = fx
        search_y = fy + int(fh * 0.45)
        search_w = fw
        search_h = int(fh * 0.5)
        
        search_x = max(0, search_x)
        search_y = max(0, search_y)
        search_w = min(img.shape[1] - search_x, search_w)
        search_h = min(img.shape[0] - search_y, search_h)
        
        roi_gray = cv2.cvtColor(img[search_y:search_y+search_h, search_x:search_x+search_w], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.equalizeHist(roi_gray)
        
        min_w = max(15, int(fw * 0.15))
        min_h = max(10, int(fh * 0.08))
        
        mouth_rects = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_w, min_h)
        )
        
        if len(mouth_rects) == 0:
            mouth_rects = mouth_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=max(scale_factor - 0.2, 1.05),
                minNeighbors=max(min_neighbors-2, 1),
                minSize=(min_w, min_h)
            )
        
        if len(mouth_rects) == 0:
            continue
        
        mouths_global = []
        for (mx, my, mw, mh) in mouth_rects:
            mouths_global.append((search_x + mx, search_y + my, mw, mh))
        
        if not detectar_todas:
            face_center_x = fx + fw/2
            best = min(mouths_global, key=lambda r: abs((r[0]+r[2]/2) - face_center_x))
            mouths_global = [best]
        
        for (x, y, w, h) in mouths_global:
            if mostrar_rectangulos:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            h_new = int(height_scale * h)
            w_new = int(width_scale * w)
            x_new = x + int(x_offset * w)
            y_new = y + int(y_offset * h)
            
            if x_new < 0 or y_new < 0 or x_new + w_new > img.shape[1] or y_new + h_new > img.shape[0]:
                continue
            
            img = aplicar_bigote_a_roi_mejorado(img, moustache, x_new, y_new, w_new, h_new)
            detected += 1
    
    return img, detected


def aplicar_bigote_a_roi_mejorado(img, moustache, x, y, w, h):
    """
    Inserta el bigote en la ROI (maneja PNG con alpha y JPG sin alpha).
    """
    try:
        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)
        
        if w <= 0 or h <= 0:
            return img
        
        frame_roi = img[y:y+h, x:x+w]
        
        # Si el bigote tiene alpha (4 canales) lo usamos directamente
        if moustache.shape[2] == 4:
            moustache_resized = cv2.resize(moustache, (w, h), interpolation=cv2.INTER_AREA)
            b, g, r, a = cv2.split(moustache_resized)
            mask = a
            mask_inv = cv2.bitwise_not(mask)
            
            moustache_bgr = cv2.merge((b, g, r))
            
            masked_moustache = cv2.bitwise_and(moustache_bgr, moustache_bgr, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            
            final = cv2.add(masked_moustache, masked_frame)
            img[y:y+h, x:x+w] = final
            return img
        
        # Si no hay alpha, generar máscara con Otsu
        moustache_resized = cv2.resize(moustache, (w, h), interpolation=cv2.INTER_AREA)
        gray_moustache = cv2.cvtColor(moustache_resized, cv2.COLOR_BGR2GRAY)
        
        _, mask = cv2.threshold(gray_moustache, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        white_ratio = np.count_nonzero(mask) / mask.size
        if white_ratio < 0.5:
            mask = cv2.bitwise_not(mask)
        
        mask_inv = cv2.bitwise_not(mask)
        
        masked_moustache = cv2.bitwise_and(moustache_resized, moustache_resized, mask=mask)
        masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        
        final = cv2.add(masked_moustache, masked_frame)
        img[y:y+h, x:x+w] = final
        
    except Exception as e:
        print(f"Error aplicando bigote: {e}")
    
    return img


def aplicar_bigote_frame(frame, moustache, mouth_cascade, scale_factor, min_neighbors):
    """Aplica bigote a un frame de video (para webcam)."""
    
    if moustache is None or mouth_cascade.empty():
        return frame, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    
    num_detections = len(mouth_rects)
    
    for (x, y, w, h) in mouth_rects:
        h_new = int(0.6 * h)
        w_new = int(1.2 * w)
        x_new = x - int(0.05 * w)
        y_new = y - int(0.55 * h)
        
        if x_new < 0 or y_new < 0 or x_new + w_new > frame.shape[1] or y_new + h_new > frame.shape[0]:
            continue
        
        frame = aplicar_bigote_a_roi_mejorado(frame, moustache, x_new, y_new, w_new, h_new)
    
    return frame, num_detections


def guardar_resultado(img, filename):
    """Guarda el resultado en disco."""
    
    output_dir = Path("output/moustache_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    try:
        cv2.imwrite(str(output_path), img)
        st.success(f"✅ Imagen guardada en: {output_path}")
        
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