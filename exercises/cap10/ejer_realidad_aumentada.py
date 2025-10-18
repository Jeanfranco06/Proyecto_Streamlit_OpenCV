import streamlit as st

# ⚠️ DEBE ser el PRIMER comando - ANTES de cualquier import que use st
st.set_page_config(page_title="Realidad Aumentada", layout="wide")

# Ahora importa TODO lo demás
import cv2
import numpy as np
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# ⚠️ IMPORTANTE: Si pose_estimation.py tiene comandos de Streamlit,
# tenemos que eliminarlos o modificarlos
try:
    # Importar sin que execute código Streamlit
    import sys
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("pose_estimation", "pose_estimation.py")
    pose_module = importlib.util.module_from_spec(spec)
    sys.modules["pose_estimation"] = pose_module
    
    # Ejecutar el módulo (pero sin st.set_page_config)
    spec.loader.exec_module(pose_module)
    
    PoseEstimator = pose_module.PoseEstimator
    ROISelector = pose_module.ROISelector
    
except Exception as e:
    st.error(f"Error importando pose_estimation: {e}")
    st.info("Alternativa: Asegúrate que pose_estimation.py NO tenga st.set_page_config()")
    st.stop()


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


def main():
    st.title("Realidad Aumentada - Detección de ROI en Streamlit")
    st.write("""
    Demostración de realidad aumentada usando OpenCV y Streamlit.

    **Instrucciones:**
    1. Inicia la cámara web.
    2. Dibuja un rectángulo directamente sobre el video para definir el ROI.
    3. Aparecerá una pirámide 3D sobre esa región.
    """)

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
        st.rerun()

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

    if CANVAS_AVAILABLE:
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

        new_roi_added = False
        if canvas_result is not None and canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            seen = st.session_state.canvas_rects_seen
            for obj in objects:
                if obj.get("type") == "rect":
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

                    if origin_x == "center":
                        left = left - width / 2.0
                    if origin_y == "center":
                        top = top - height / 2.0

                    left = int(round(left))
                    top = int(round(top))

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

        if new_roi_added:
            st.rerun()

    st.image(processed_frame_rgb, channels="RGB", caption="Video con ROI y pirámide")


if __name__ == '__main__':
    main()