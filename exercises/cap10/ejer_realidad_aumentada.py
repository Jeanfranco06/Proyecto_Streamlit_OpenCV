import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebrtcMode, RTCConfiguration
import av
import cv2
from PIL import Image

# Importa tu clase Tracker
from .pose_estimation import PoseEstimator, ROISelector

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


class VideoProcessor:
    """Procesa frames de video en tiempo real"""
    def __init__(self, tracker):
        self.tracker = tracker

    def recv(self, frame):
        # Convierte frame de av.VideoFrame a numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Procesa con tu Tracker
        processed = self.tracker.process_frame(img)
        
        # Convierte de vuelta a av.VideoFrame
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


def run():
    st.title("Realidad Aumentada - WebRTC Streaming")
    st.write("""
    Demostración de realidad aumentada usando OpenCV y Streamlit WebRTC.

    **Instrucciones:**
    1. Haz clic en **'Start'** para iniciar la cámara
    2. Dibuja un rectángulo sobre el video para definir el ROI
    3. Aparecerá una pirámide 3D sobre esa región
    4. Puedes agregar más de una región
    """)

    # Inicializa el tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = Tracker(scaling_factor=0.8)

    tracker = st.session_state.tracker

    # Configuración de WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Crea el streamer de WebRTC
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

    # Procesa frames si el stream está activo
    if webrtc_ctx.state.playing:
        video_processor = VideoProcessor(tracker)
        webrtc_ctx.rtc_peer_connection.video_processor = video_processor

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpiar Selecciones"):
                tracker.tracker.clear_targets()
                tracker.rect = None
                st.rerun()

        st.success("✓ Stream activo - Cámara conectada")
    else:
        st.info("Haz clic en 'Start' para iniciar la cámara")


if __name__ == '__main__':
    run()