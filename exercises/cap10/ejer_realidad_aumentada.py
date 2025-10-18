import streamlit as st
import streamlit as st

# ⚠️ DEBE SER EL PRIMER COMANDO STREAMLIT
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

# Importa pose_estimation (está en el mismo directorio)
try:
    from pose_estimation import PoseEstimator, ROISelector
except ImportError:
    # Si no está disponible, usa las definiciones incorporadas
    st.warning("⚠️ pose_estimation.py no encontrado, usando versión embedida")
    
    import cv2
    from dataclasses import dataclass
    from collections import namedtuple
    
    @dataclass
    class Target:
        rect: tuple
        kp: list
        des: np.ndarray
        quad: np.ndarray

    @dataclass
    class TrackedItem:
        target: Target
        quad: np.ndarray
        points_cur: np.ndarray

    class ROISelector:
        def __init__(self, win_name, frame, on_select_callback):
            self.win_name = win_name
            self.on_select_callback = on_select_callback
            self.frame_h, self.frame_w = (frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)
            self.dragging = False
            self.x0 = self.y0 = self.x1 = self.y1 = 0
            self.current_rect = None
            self.gui_available = True
            try:
                cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(self.win_name, self._on_mouse)
            except Exception:
                self.gui_available = False

        def _on_mouse(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.x0, self.y0 = x, y
                self.x1, self.y1 = x, y
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.x1, self.y1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
                x0, y0 = min(self.x0, self.x1), min(self.y0, self.y1)
                x1, y1 = max(self.x0, self.x1), max(self.y0, self.y1)
                if (x1 - x0) >= 10 and (y1 - y0) >= 10:
                    rect = (x0, y0, x1, y1)
                    self.current_rect = rect
                    if self.on_select_callback is not None:
                        self.on_select_callback(rect)

        def draw_rect(self, img, rect):
            if rect is not None:
                x0, y0, x1, y1 = rect
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
            if self.dragging:
                x0, y0 = self.x0, self.y0
                x1, y1 = self.x1, self.y1
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)

    class PoseEstimator(object):
        def __init__(self, nfeatures=1000, min_matches=10):
            self.cur_target = namedtuple('Current', 'image rect keypoints descriptors data')
            self.tracked_target = namedtuple('Tracked', 'target points_prev points_cur H quad')
            self.min_matches = min_matches
            self.feature_detector = cv2.ORB_create(nfeatures=nfeatures)
            self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.tracking_targets = []

        def detect_features(self, frame):
            if frame is None:
                return [], np.array([], dtype=np.uint8)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            kps, des = self.feature_detector.detectAndCompute(gray, None)
            if des is None:
                des = np.array([], dtype=np.uint8)
            return kps, des

        def add_target(self, image, rect, data=None):
            x0, y0, x1, y1 = rect
            kps, des = self.detect_features(image)
            if len(kps) == 0:
                return
            kept_kps = []
            kept_desc = []
            for i, kp in enumerate(kps):
                x, y = kp.pt
                if x0 <= int(x) <= x1 and y0 <= int(y) <= y1:
                    kept_kps.append(kp)
                    if des is not None and len(des) > i:
                        kept_desc.append(des[i])
            if len(kept_kps) < self.min_matches:
                return
            kept_desc = np.asarray(kept_desc, dtype=np.uint8)
            target = self.cur_target(image=image, rect=rect, keypoints=kept_kps,
                                     descriptors=kept_desc, data=data)
            self.tracking_targets.append(target)

        def track_target(self, frame):
            if len(self.tracking_targets) == 0:
                return []
            cur_kps, cur_des = self.detect_features(frame)
            if cur_des is None or len(cur_kps) < self.min_matches:
                return []
            tracked = []
            for target in self.tracking_targets:
                if target.descriptors is None or len(target.descriptors) == 0:
                    continue
                try:
                    matches = self.feature_matcher.knnMatch(target.descriptors, cur_des, k=2)
                except Exception:
                    continue
                good = []
                for m in matches:
                    if len(m) < 2:
                        continue
                    m1, m2 = m
                    if m1.distance < 0.75 * m2.distance:
                        good.append(m1)
                if len(good) < self.min_matches:
                    continue
                pts_prev = np.float32([ target.keypoints[m.queryIdx].pt for m in good ])
                pts_cur = np.float32([ cur_kps[m.trainIdx].pt for m in good ])
                if len(pts_prev) < 4 or len(pts_cur) < 4:
                    continue
                H, mask = cv2.findHomography(pts_prev.reshape(-1,1,2), pts_cur.reshape(-1,1,2), cv2.RANSAC, 3.0)
                if H is None:
                    continue
                mask = mask.ravel().astype(bool)
                if mask.sum() < self.min_matches:
                    continue
                pts_prev_in = pts_prev[mask]
                pts_cur_in = pts_cur[mask]
                x0, y0, x1, y1 = target.rect
                quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1,1,2)
                quad_proj = cv2.perspectiveTransform(quad, H).reshape(-1,2)
                track = self.tracked_target(target=target,
                                           points_prev=pts_prev_in,
                                           points_cur=pts_cur_in,
                                           H=H,
                                           quad=quad_proj)
                tracked.append(track)
            tracked.sort(key=lambda t: len(t.points_prev), reverse=True)
            return tracked

        def clear_targets(self):
            self.tracking_targets = []
            self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


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
    4. Puedes agregar más de una región.
    5. Usa *Limpiar Selecciones* para reiniciar.
    """)

    # Estado persistente
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
    else:
        st.warning("⚠️ streamlit-drawable-canvas no disponible.")

    st.image(processed_frame_rgb, channels="RGB", caption="Video con ROI y pirámide")


if __name__ == '__main__':
    main()
# ⚠️ DEBE ser el PRIMER comando - ANTES de cualquier import que use st

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