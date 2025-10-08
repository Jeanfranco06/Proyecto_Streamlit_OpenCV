# pose_estimation.py
import cv2
import numpy as np
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class Target:
    rect: tuple          # (x0, y0, x1, y1)
    kp: list             # keypoints absolutos (coordenadas en frame) como np.float32 Nx2
    des: np.ndarray      # descriptores ORB
    quad: np.ndarray     # esquinas del rectángulo en el frame original (4x2 float32)

@dataclass
class TrackedItem:
    target: Target
    quad: np.ndarray       # esquinas proyectadas (4x2 float32) en el frame actual
    points_cur: np.ndarray # puntos coincidentes en el frame actual (Mx2 float32)


class ROISelector:
    """
    Selecciona ROI con el mouse:
    - Click y arrastrar para dibujar.
    - Al soltar, llama al callback(rect).
    """
    def __init__(self, win_name, frame, on_select_callback):
        self.win_name = win_name
        self.on_select_callback = on_select_callback
        self.frame_h, self.frame_w = (frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)

        self.dragging = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0
        self.current_rect = None

        # Algunos paquetes de OpenCV (p. ej. opencv-python-headless) no incluyen
        # soporte para funciones de ventana (namedWindow, setMouseCallback).
        # Intentamos crear la ventana y asociar el callback; si falla, marcamos
        # que la GUI no está disponible para evitar que la app se rompa.
        self.gui_available = True
        try:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.win_name, self._on_mouse)
        except Exception:
            # No podemos crear ventanas en este entorno (headless). La
            # selección con el ratón no funcionará; la app debe proveer
            # otra forma de seleccionar ROI (por ejemplo, controles de Streamlit).
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
            # Evita rectángulos demasiado pequeños
            if (x1 - x0) >= 10 and (y1 - y0) >= 10:
                rect = (x0, y0, x1, y1)
                self.current_rect = rect
                if self.on_select_callback is not None:
                    self.on_select_callback(rect)

    def draw_rect(self, img, rect):
        if rect is not None:
            x0, y0, x1, y1 = rect
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
        # Si estás arrastrando, dibuja el rect provisional
        if self.dragging:
            x0, y0 = self.x0, self.y0
            x1, y1 = self.x1, self.y1
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)


class PoseEstimator(object):
    def __init__(self, nfeatures=1000, min_matches=10):
        # Named tuples para targets y objetos trackeados
        self.cur_target = namedtuple('Current', 'image rect keypoints descriptors data')
        self.tracked_target = namedtuple('Tracked', 'target points_prev points_cur H quad')

        # Parámetros
        self.min_matches = min_matches
        # ORB detector
        self.feature_detector = cv2.ORB_create(nfeatures=nfeatures)
        # Usamos BFMatcher con Hamming para descriptores ORB (binarios)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Lista de objetivos registrados (cada uno contiene keypoints y descriptores)
        self.tracking_targets = []

    def detect_features(self, frame):
        # Acepta BGR o gray. ORB requiere gray.
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
        """
        Registra un nuevo target a partir de la imagen completa y el rect (x0,y0,x1,y1).
        Solo se guardan keypoints/descritores dentro del rect.
        """
        x0, y0, x1, y1 = rect
        kps, des = self.detect_features(image)
        if len(kps) == 0:
            return  # nada que añadir

        # Filtrar keypoints y descriptores por la ROI
        kept_kps = []
        kept_desc = []
        for i, kp in enumerate(kps):
            x, y = kp.pt
            if x0 <= int(x) <= x1 and y0 <= int(y) <= y1:
                kept_kps.append(kp)
                # des puede ser array vacío; proteger
                if des is not None and len(des) > i:
                    kept_desc.append(des[i])

        if len(kept_kps) < self.min_matches:
            # No hay suficientes features en el ROI para ser un buen target
            return

        kept_desc = np.asarray(kept_desc, dtype=np.uint8)
        target = self.cur_target(image=image, rect=rect, keypoints=kept_kps,
                                 descriptors=kept_desc, data=data)
        self.tracking_targets.append(target)

    def track_target(self, frame):
        """
        Trackea todos los targets registrados en el frame actual.
        Devuelve lista de tracked_target (target, points_prev, points_cur, H, quad).
        """
        if len(self.tracking_targets) == 0:
            return []

        cur_kps, cur_des = self.detect_features(frame)
        if cur_des is None or len(cur_kps) < self.min_matches:
            return []

        tracked = []

        # Para cada target hacemos knnMatch target.des -> cur_des
        for target in self.tracking_targets:
            if target.descriptors is None or len(target.descriptors) == 0:
                continue
            try:
                matches = self.feature_matcher.knnMatch(target.descriptors, cur_des, k=2)
            except Exception:
                continue

            # Ratio test (Lowe)
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
        # Recreate matcher to ensure clean state
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)