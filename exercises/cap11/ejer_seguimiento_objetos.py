import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from typing import Tuple, Optional
from PIL import Image


class ObjectTracker:
    TRACKING_METHODS = {
        'CSRT': 'CSRT',
        'KCF': 'KCF',
        'MIL': 'MIL'
    }

    def __init__(self):
        self.tracker = None
        self.bbox = None
        self.tracking_method = 'CSRT'
        self.success = False

    def init_tracker(self, method: str):
        """Inicializa el tracker con el método especificado"""
        self.tracking_method = method
        if method == 'CSRT':
            self.tracker = cv2.legacy.TrackerCSRT_create()
        elif method == 'KCF':
            self.tracker = cv2.legacy.TrackerKCF_create()
        elif method == 'MIL':
            self.tracker = cv2.legacy.TrackerMIL_create()
        else:
            raise ValueError(f"Método de tracking desconocido: {method}")

    def start_tracking(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Inicia el seguimiento del objeto"""
        self.bbox = bbox
        return self.tracker.init(frame, bbox)

    def update_tracking(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Actualiza la posición del objeto"""
        if self.tracker is None:
            return False, None
        self.success, bbox = self.tracker.update(frame)
        if self.success:
            self.bbox = tuple(map(int, bbox))
        return self.success, self.bbox

    def draw_bbox(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja el bounding box"""
        if self.success and self.bbox is not None:
            x, y, w, h = self.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking: {self.tracking_method}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def run():
    st.title("🎥 Seguimiento de Objetos en Tiempo Real")
    st.write("""
    Este ejercicio demuestra el seguimiento de objetos en tiempo real usando algoritmos de OpenCV.

    **Instrucciones:**
    1. Activa la cámara
    2. Dibuja un rectángulo sobre el objeto que quieras seguir
    3. Observa cómo el sistema sigue al objeto
    4. Puedes reiniciar la selección en cualquier momento
    """)

    # Selección del método de tracking
    tracking_method = st.selectbox(
        'Selecciona el método de tracking',
        list(ObjectTracker.TRACKING_METHODS.keys()),
        index=0
    )

    # Inicializar estado de sesión
    if 'tracker' not in st.session_state:
        st.session_state.tracker = ObjectTracker()
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'bbox' not in st.session_state:
        st.session_state.bbox = None

    # Botón para reiniciar tracking
    if st.button("🔁 Reiniciar Tracking"):
        st.session_state.tracker = ObjectTracker()
        st.session_state.bbox = None

    # Checkbox para iniciar cámara
    camera_on = st.checkbox('📸 Iniciar Cámara', value=st.session_state.camera_on)

    if camera_on:
        cap = cv2.VideoCapture(0)
        video_placeholder = st.empty()
        canvas_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo acceder a la cámara")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Si aún no se seleccionó ROI → permitir dibujar
            if st.session_state.bbox is None:
                st.info("Dibuja un rectángulo sobre el objeto a seguir")

                # Canvas para dibujar ROI
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.3)",
                    stroke_width=2,
                    stroke_color="#00FF00",
                    background_image=Image.fromarray(frame_rgb),
                    update_streamlit=True,
                    height=480,
                    width=640,
                    drawing_mode="rect",
                    key="roi_canvas"
                )

                # Procesar el rectángulo dibujado
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if len(objects) > 0:
                        obj = objects[0]
                        bbox = (
                            int(obj["left"]),
                            int(obj["top"]),
                            int(obj["width"]),
                            int(obj["height"])
                        )
                        st.session_state.bbox = bbox
                        st.session_state.tracker.init_tracker(tracking_method)
                        st.session_state.tracker.start_tracking(frame, bbox)
                        st.success("✅ Objeto seleccionado correctamente")
                        st.experimental_rerun()
                break  # detener el bucle hasta seleccionar ROI

            else:
                # Actualizar seguimiento
                success, bbox = st.session_state.tracker.update_tracking(frame)
                frame_tracked = st.session_state.tracker.draw_bbox(frame.copy())
                frame_rgb = cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

    st.session_state.camera_on = camera_on
    
if __name__ == '__main__':
    run()