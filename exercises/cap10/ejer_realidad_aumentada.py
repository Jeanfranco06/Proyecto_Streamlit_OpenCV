import streamlit as st
st.set_page_config(page_title="Realidad Aumentada", layout="wide")

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from collections import namedtuple
import av
import threading

try:
    from streamlit_webrtc import webrtc_streamer, WebrtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.error("❌ streamlit-webrtc no instalado. Instala: pip install streamlit-webrtc")
    st.stop()


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
        except:
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


class PoseEstimator:
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
            except:
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
            pts_prev = np.float32([target.keypoints[m.queryIdx].pt for m in good])
            pts_cur = np.float32([cur_kps[m.trainIdx].pt for m in good])
            if len(pts_prev) < 4 or len(pts_cur) < 4:
                continue
            H, mask = cv2.findHomography(pts_prev.reshape(-1, 1, 2), pts_cur.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
            if H is None:
                continue
            mask = mask.ravel().astype(bool)
            if mask.sum() < self.min_matches:
                continue
            pts_prev_in = pts_prev[mask]
            pts_cur_in = pts_cur[mask]
            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
            quad_proj = cv2.perspectiveTransform(quad, H).reshape(-1, 2)
            track = self.tracked_target(target=target, points_prev=pts_prev_in,
                                       points_cur=pts_cur_in, H=H, quad=quad_proj)
            tracked.append(track)
        tracked.sort(key=lambda t: len(t.points_prev), reverse=True)
        return tracked

    def clear_targets(self):
        self.tracking_targets = []
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


class Tracker:
    def __init__(self, scaling_factor=0.8):
        self.rect = None
        self.scaling_factor = scaling_factor
        self.tracker = PoseEstimator()
        self.first_frame = True
        self.frame = None
        self.roi_selector = None
        self.overlay_vertices = np.float32([
            [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0.5, 0.5, 4]
        ])
        self.overlay_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
        self.color_base = (0, 255, 0)
        self.color_lines = (0, 0, 0)

    def set_rect(self, rect):
        self.rect = rect
        self.tracker.add_target(self.frame, rect)

    def process_frame(self, frame):
        frame = cv2.resize(frame, None, fx=self.scaling_factor, fy=self.scaling_factor, 
                          interpolation=cv2.INTER_AREA)
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
        K = np.float64([[w, 0, 0.5*(w-1)], [0, w, 0.5*(h-1)], [0, 0, 1.0]])
        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(objectPoints=quad_3d, imagePoints=tracked.quad,
                                      cameraMatrix=K, distCoeffs=dist_coef)
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


class VideoProcessor:
    def __init__(self, tracker):
        self.tracker = tracker

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = self.tracker.process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


st.title("Realidad Aumentada con WebRTC")
st.write("✅ Funciona en Streamlit Cloud con cámara en vivo")

if 'tracker' not in st.session_state:
    st.session_state.tracker = Tracker(scaling_factor=0.8)

tracker = st.session_state.tracker

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="ar-tracking",
    mode=WebrtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}},
        "audio": False
    },
    async_processing=True,
)

if webrtc_ctx.state.playing:
    video_processor = VideoProcessor(tracker)
    webrtc_ctx.video_processor = video_processor
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Limpiar ROI"):
            tracker.tracker.clear_targets()
            tracker.rect = None
            st.rerun()
    
    st.success("✓ Stream activo - Cámara conectada")
    st.info("Dibuja rectángulos en la cámara para definir ROI")
else:
    st.info("Haz clic en 'Start' para iniciar la cámara")