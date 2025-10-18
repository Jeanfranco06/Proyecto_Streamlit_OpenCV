# ar_ejercicio10.py
import cv2
import numpy as np
import streamlit as st
from PIL import Image
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

from .pose_estimation import PoseEstimator, ROISelector
from streamlit.components.v1 import html as components_html
import base64

class Tracker(object):
    def __init__(self, scaling_factor=0.8):
        self.rect = None
        self.scaling_factor = scaling_factor
        self.tracker = PoseEstimator()
        self.first_frame = True
        self.frame = None
        self.roi_selector = None

        self.overlay_vertices = np.float32([
            [0,   0,   0],
            [0,   1,   0],
            [1,   1,   0],
            [1,   0,   0],
            [0.5, 0.5, 4]
        ])
        self.overlay_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]

        self.color_base = (0, 255, 0)
        self.color_lines = (0, 0, 0)

    def set_rect(self, rect):
        self.rect = rect
        # add_target espera coordenadas en la imagen que usamos en self.frame
        self.tracker.add_target(self.frame, rect)

    def process_frame(self, frame):
        frame = cv2.resize(frame, None, fx=self.scaling_factor,
                          fy=self.scaling_factor, interpolation=cv2.INTER_AREA)

        if self.first_frame:
            self.frame = frame.copy()
            self.roi_selector = ROISelector("AR", frame, self.set_rect)
            self.first_frame = False

        self.frame = frame.copy()
        img = frame.copy()

        tracked = self.tracker.track_target(self.frame)
        for item in tracked:
            cv2.polylines(img, [np.int32(item.quad)], True, self.color_lines, 2)
            for (x, y) in np.int32(item.points_cur):
                cv2.circle(img, (x, y), 2, self.color_lines)
            self.overlay_graphics(img, item)

        if self.roi_selector:
            self.roi_selector.draw_rect(img, self.rect)

        return img

    def overlay_graphics(self, img, tracked):
        x_start, y_start, x_end, y_end = tracked.target.rect
        quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0],
                             [x_end, y_end, 0], [x_start, y_end, 0]])

        h, w = img.shape[:2]
        K = np.float64([[w, 0, 0.5*(w-1)],
                       [0, w, 0.5*(h-1)],
                       [0, 0, 1.0]])
        dist_coef = np.zeros(4)

        ret, rvec, tvec = cv2.solvePnP(objectPoints=quad_3d,
                                      imagePoints=tracked.quad,
                                      cameraMatrix=K,
                                      distCoeffs=dist_coef)
        if not ret:
            return

        verts = self.overlay_vertices * [(x_end-x_start), (y_end-y_start), -(x_end-x_start)*0.3] + (x_start, y_start, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, cameraMatrix=K, distCoeffs=dist_coef)[0].reshape(-1, 2)
        verts_floor = np.int32(verts).reshape(-1, 2)

        cv2.drawContours(img, contours=[verts_floor[:4]], contourIdx=-1, color=self.color_base, thickness=-3)
        cv2.drawContours(img, contours=[np.vstack((verts_floor[:2], verts_floor[4:5]))], contourIdx=-1, color=(0,255,0), thickness=-3)
        cv2.drawContours(img, contours=[np.vstack((verts_floor[1:3], verts_floor[4:5]))], contourIdx=-1, color=(255,0,0), thickness=-3)
        cv2.drawContours(img, contours=[np.vstack((verts_floor[2:4], verts_floor[4:5]))], contourIdx=-1, color=(0,0,150), thickness=-3)
        cv2.drawContours(img, contours=[np.vstack((verts_floor[3:4], verts_floor[0:1], verts_floor[4:5]))], contourIdx=-1, color=(255,255,0), thickness=-3)

        for i, j in self.overlay_edges:
            (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
            cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), self.color_lines, 2)


def run():
    st.title("Realidad Aumentada - Detección de ROI en Streamlit")
    st.write("""
    Demostración de realidad aumentada usando OpenCV y Streamlit.

    **Instrucciones:**
    1. Inicia la cámara web.
    2. Dibuja un rectángulo directamente sobre el video para definir el ROI.
    3. Aparecerá una pirámide 3D sobre esa región.
    4. Puedes agregar más de una región.
    5. Usa *Limpiar Selecciones* para reiniciar.
    """)

    # --- Estado persistente seguro ---
    # Si existe otro tipo de tracker guardado (p. ej. ObjectTracker viejo), lo forzamos a crear de nuevo
    if 'tracker' in st.session_state:
        if type(st.session_state.tracker).__name__ != "Tracker":
            del st.session_state['tracker']

    if 'tracker' not in st.session_state:
        st.session_state.tracker = Tracker(scaling_factor=0.8)

    if 'canvas_rects_seen' not in st.session_state:
        st.session_state.canvas_rects_seen = []

    if 'camera' not in st.session_state or st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)

    tracker = st.session_state.tracker
    cap = st.session_state.camera

    col1, col2 = st.columns(2)
    with col1:
        limpiar = st.button("Limpiar Selecciones")
    with col2:
        detener = st.button("Detener Cámara")

    if limpiar:
        tracker.tracker.clear_targets()
        tracker.rect = None
        st.session_state.canvas_rects_seen = []
        st.experimental_rerun()

    if detener:
        if cap is not None:
            cap.release()
        st.session_state.camera = None
        st.warning("Cámara detenida.")
        return

    ret, frame = cap.read()
    if not ret:
        st.error("No se pudo acceder a la cámara. Verifica que esté conectada.")
        return

    processed_frame = tracker.process_frame(frame)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(processed_frame_rgb)

    st.subheader("Selecciona ROI directamente sobre el video")

    # Dibuja canvas con el tamaño exacto de la imagen para minimizar escalados CSS
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.0)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=pil_img,
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="rect",
        key="roi_canvas_main",
    )

    # --- PROCESAMIENTO ROBUSTO DE OBJETOS DEVUELTOS POR FABRIC.JS ---
    new_roi_added = False
    if canvas_result is not None and canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        seen = st.session_state.canvas_rects_seen
        for obj in objects:
            if obj.get("type") == "rect":
                # width/height pueden estar escalados por scaleX/scaleY
                base_w = obj.get("width", 0)
                base_h = obj.get("height", 0)
                scale_x = obj.get("scaleX", 1)
                scale_y = obj.get("scaleY", 1)
                width = int(round(base_w * scale_x))
                height = int(round(base_h * scale_y))

                origin_x = obj.get("originX", "left")
                origin_y = obj.get("originY", "top")
                left = obj.get("left", 0)
                top = obj.get("top", 0)

                # Si el objeto usa centro como origen, ajustamos
                if origin_x == "center":
                    left = left - width / 2.0
                if origin_y == "center":
                    top = top - height / 2.0

                left = int(round(left))
                top = int(round(top))

                # clamp a los límites de la imagen
                left = max(0, min(left, pil_img.width - 1))
                top = max(0, min(top, pil_img.height - 1))
                width = max(1, min(width, pil_img.width - left))
                height = max(1, min(height, pil_img.height - top))

                key_tuple = (left, top, width, height)
                if key_tuple not in seen:
                    rect = (left, top, left + width, top + height)
                    tracker.set_rect(rect)
                    seen.append(key_tuple)
                    new_roi_added = True
                    st.success(f"ROI añadida: {rect}")

    # Mostrar frame con overlay
    st.image(processed_frame_rgb, channels="RGB", caption="Video con ROI y pirámide")

    # Sólo rerun si añadimos un ROI (evita reruns infinitos)
    if new_roi_added:
        st.experimental_rerun()

# SOLUCIÓN 1: Usar streamlit-webrtc (Recomendado para Streamlit en navegador)
# Instala: pip install streamlit-webrtc

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2

def run_with_webrtc():
    st.title("Realidad Aumentada con WebRTC")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Aquí procesas con tu Tracker
            processed = tracker.process_frame(img)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="AR-tracking",
        mode=WebrtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True},
        async_processing=True,
    )


# SOLUCIÓN 2: Diagnosticar el problema (ejecuta esto primero)
import cv2
import sys

def diagnosticar_camara():
    print("=== DIAGNÓSTICO DE CÁMARA ===")
    print(f"Sistema: {sys.platform}")
    print(f"OpenCV versión: {cv2.__version__}")
    
    # Intenta acceder a la cámara
    cap = cv2.VideoCapture(0)
    print(f"VideoCapture(0) abierto: {cap.isOpened()}")
    
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Frame capturado: {ret}")
        if ret:
            print(f"Dimensiones: {frame.shape}")
        cap.release()
    else:
        print("❌ No se pudo abrir la cámara")
        print("\nIntentando otras cámaras...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✓ Cámara {i} disponible")
                cap.release()

diagnosticar_camara()


# SOLUCIÓN 3: Usar archivo de video o imagen estática para testing
def run_con_archivo():
    st.title("Realidad Aumentada - Modo Archivo")
    
    opcion = st.radio("Selecciona entrada:", ["Cámara", "Video", "Imagen"])
    
    if opcion == "Cámara":
        cap = cv2.VideoCapture(0)
    elif opcion == "Video":
        video_file = st.file_uploader("Sube un video", type=['mp4', 'avi'])
        if video_file:
            cap = cv2.VideoCapture(video_file.name)
    else:
        imagen = st.file_uploader("Sube una imagen", type=['jpg', 'png'])
        if imagen:
            from PIL import Image
            img = Image.open(imagen)
            # Procesar imagen única
            return
    
    if cap and cap.isOpened():
        ret, frame = cap.read()
        # ... resto del código


# SOLUCIÓN 4: Modificar tu código para mejor manejo de errores
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def run_mejorado():
    st.title("Realidad Aumentada - Detección de ROI")
    
    # Intentar diferentes índices de cámara
    cap = None
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            st.info(f"✓ Cámara encontrada en índice {camera_index}")
            break
        cap.release()
    
    if cap is None or not cap.isOpened():
        st.error("""
        ❌ No se pudo acceder a la cámara.
        
        **Posibles soluciones:**
        1. **Verifica permisos**: La app necesita permiso para usar la cámara
        2. **Cámara conectada**: Asegúrate que esté enchufada
        3. **Otra app la usa**: Cierra otras apps que usen la cámara
        4. **Prueba con HTTPS**: Streamlit Cloud requiere HTTPS
        5. **Usa streamlit-webrtc**: Mejor opción para navegador
        
        **Para desarrollo local:**
        ```
        streamlit run app.py
        ```
        
        **Para Streamlit Cloud:**
        Usa `streamlit-webrtc` en lugar de `cv2.VideoCapture`
        """)
        return
    
    # ... resto de tu código con mejor manejo


# SOLUCIÓN 5: Si estás en Streamlit Cloud, esta es la ÚNICA opción viable
# Reemplaza tu código con streamlit-webrtc:

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebrtcMode, RTCConfiguration
import av

def run_streamlit_cloud():
    """Solución para Streamlit Cloud (HTTPS obligatorio)"""
    st.title("Realidad Aumentada en Streamlit Cloud")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    class VideoProcessor:
        def __init__(self, tracker):
            self.tracker = tracker
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed = self.tracker.process_frame(img)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="ar-tracking",
        mode=WebrtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.success("Streaming activo")

if __name__ == '__main__':
    run_mejorado()
