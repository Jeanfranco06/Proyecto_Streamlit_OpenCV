import streamlit as st

# DEBE SER EL PRIMER COMANDO
st.set_page_config(page_title="Realidad Aumentada", layout="wide")

import cv2
import numpy as np
from PIL import Image
import threading
import time

# Importa tu clase Tracker
# from .pose_estimation import PoseEstimator, ROISelector

class Tracker(object):
    def __init__(self, scaling_factor=0.8):
        self.rect = None
        self.scaling_factor = scaling_factor
        self.first_frame = True
        self.frame = None
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

    def process_frame(self, frame):
        frame = cv2.resize(frame, None, fx=self.scaling_factor,
                          fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
        self.frame = frame.copy()
        return frame


def find_camera():
    """Busca la primera cámara disponible"""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None


def run():
    st.set_page_config(page_title="Realidad Aumentada", layout="wide")
    st.title("Realidad Aumentada - OpenCV")
    st.write("""
    Demostración de realidad aumentada usando OpenCV y Streamlit.
    
    **Funciona en desarrollo local sin WebRTC**
    """)

    # Inicializa sesión
    if 'tracker' not in st.session_state:
        st.session_state.tracker = Tracker(scaling_factor=0.8)
    
    if 'capturing' not in st.session_state:
        st.session_state.capturing = False
    
    if 'frame_placeholder' not in st.session_state:
        st.session_state.frame_placeholder = None

    tracker = st.session_state.tracker

    # Controles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start = st.button("▶️ Iniciar Cámara", key="start_btn")
    with col2:
        stop = st.button("⏹️ Detener Cámara", key="stop_btn")
    with col3:
        clear = st.button("🔄 Limpiar", key="clear_btn")

    if start:
        st.session_state.capturing = True
        st.rerun()
    
    if stop:
        st.session_state.capturing = False
        st.rerun()
    
    if clear:
        st.session_state.tracker = Tracker(scaling_factor=0.8)
        st.rerun()

    # Placeholder para mostrar video
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.capturing:
        # Busca cámara disponible
        camera_idx = find_camera()
        
        if camera_idx is None:
            st.error("""
            ❌ No se encontró ninguna cámara disponible.
            
            **Soluciones:**
            - Verifica que la cámara esté conectada
            - Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc)
            - En Windows/Mac, verifica permisos de cámara
            - Intenta reconectar la cámara USB
            """)
            st.session_state.capturing = False
        else:
            cap = cv2.VideoCapture(camera_idx)
            
            if not cap.isOpened():
                st.error(f"No se pudo abrir la cámara en índice {camera_idx}")
                st.session_state.capturing = False
            else:
                # Configurar propiedades de cámara
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                status_placeholder.info(f"✓ Cámara abierta (índice: {camera_idx})")
                
                frame_count = 0
                while st.session_state.capturing:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Error al capturar frame")
                        break
                    
                    # Procesa frame con tracker
                    processed = tracker.process_frame(frame)
                    
                    # Convierte BGR a RGB para mostrar
                    frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    
                    # Muestra el frame
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    frame_count += 1
                    status_placeholder.metric("Frames", frame_count)
                    
                    # Control de velocidad
                    time.sleep(0.01)
                
                cap.release()
                status_placeholder.success("Cámara cerrada")
    else:
        # Muestra una imagen de espera
        placeholder_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder_img, "Presiona 'Iniciar Camara' para comenzar", 
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frame_placeholder.image(placeholder_img, channels="RGB", use_column_width=True)


if __name__ == '__main__':
    run()