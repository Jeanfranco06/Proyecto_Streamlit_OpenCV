"""
Configuración global del proyecto OpenCV Interactive Tutorial
"""
import os
from pathlib import Path

# ============================================
# INFORMACIÓN DE LA APLICACIÓN
# ============================================
APP_TITLE = "LWJeac OpenCV"
APP_ICON = ""
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Tutorial interactivo de OpenCV con Streamlit"

# ============================================
# CONFIGURACIÓN DE PÁGINA STREAMLIT
# ============================================
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': None,
        'Report a bug': None,
        'About': f"# {APP_TITLE}\nVersión {APP_VERSION}\n\n{APP_DESCRIPTION}"
    }
}

# ============================================
# RUTAS DEL PROYECTO
# ============================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "output"
EXERCISES_DIR = BASE_DIR / "exercises"

# Crear directorios si no existen
for directory in [DATA_DIR, IMAGES_DIR, VIDEOS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# DEFINICIÓN DE CAPÍTULOS Y EJERCICIOS
# ============================================
CHAPTERS = {
    "chapter1": {
        "title": "📖 Capítulo 1: Fundamentos",
        "description": "Introducción a OpenCV: lectura, escritura y operaciones básicas",
        "exercises": {
            "ex01_basic_read": "Lectura y visualización de imágenes",
            "ex02_resize": "Redimensionamiento de imágenes",
            "ex03_crop": "Recorte de imágenes",
            "ex04_color_spaces": "Espacios de color (BGR, RGB, HSV, LAB)",
            "ex05_drawing": "Dibujo de formas básicas",
        }
    },
    "chapter2": {
        "title": "🎨 Capítulo 2: Procesamiento de Imágenes",
        "description": "Filtros, transformaciones y mejoras de imagen",
        "exercises": {
            "ex01_threshold": "Umbralización (Simple, Otsu, Adaptativa)",
            "ex02_blur": "Filtros de suavizado (Gaussian, Median, Bilateral)",
            "ex03_edge_detection": "Detección de bordes (Canny, Sobel, Laplacian)",
            "ex04_morphology": "Operaciones morfológicas (Erosión, Dilatación)",
            "ex05_histogram": "Histogramas y ecualización",
        }
    },
    "chapter3": {
        "title": "🔍 Capítulo 3: Detección de Características",
        "description": "Contornos, formas y características clave",
        "exercises": {
            "ex01_contours": "Detección de contornos",
            "ex02_shape_detection": "Detección de formas geométricas",
            "ex03_corner_detection": "Detección de esquinas (Harris, Shi-Tomasi)",
            "ex04_template_matching": "Template Matching",
            "ex05_feature_matching": "Correspondencia de características (SIFT, ORB)",
        }
    },
    "chapter4": {
        "title": "📹 Capítulo 4: Procesamiento de Video",
        "description": "Captura, procesamiento y análisis de video",
        "exercises": {
            "ex01_video_read": "Lectura y procesamiento de video",
            "ex02_background_subtraction": "Substracción de fondo",
            "ex03_motion_detection": "Detección de movimiento",
            "ex04_optical_flow": "Flujo óptico (Dense y Sparse)",
            "ex05_object_tracking": "Seguimiento de objetos",
        }
    },
    "chapter5": {
        "title": "🤖 Capítulo 5: Visión Computacional Avanzada",
        "description": "Reconocimiento facial, detección de objetos y más",
        "exercises": {
            "ex01_face_detection": "Detección facial (Haar Cascades)",
            "ex02_face_recognition": "Reconocimiento facial",
            "ex03_object_detection": "Detección de objetos",
            "ex04_pose_estimation": "Estimación de pose",
            "ex05_image_segmentation": "Segmentación de imágenes",
        }
    }
}

# ============================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================
DEFAULT_IMAGE_WIDTH = 600
MAX_IMAGE_SIZE = (1920, 1080)
THUMBNAIL_SIZE = (200, 200)

# ============================================
# COLORES (BGR para OpenCV)
# ============================================
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "gray": (128, 128, 128),
}

# ============================================
# TEMA DE COLORES PARA UI
# ============================================
UI_COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff9800",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}

# ============================================
# CONFIGURACIÓN DE PROCESAMIENTO
# ============================================
# Tamaños de kernel válidos para diferentes operaciones
KERNEL_SIZES = {
    "blur": [3, 5, 7, 9, 11, 15, 21, 31],
    "morphology": [3, 5, 7, 9, 11, 13, 15],
    "edge": [3, 5, 7],
}

# Métodos de interpolación
INTERPOLATION_METHODS = {
    "Nearest": 0,  # cv2.INTER_NEAREST
    "Linear": 1,   # cv2.INTER_LINEAR
    "Cubic": 2,    # cv2.INTER_CUBIC
    "Area": 3,     # cv2.INTER_AREA
    "Lanczos": 4,  # cv2.INTER_LANCZOS4
}

# Formatos de imagen soportados
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

# ============================================
# CONFIGURACIÓN DE PERFORMANCE
# ============================================
# Número máximo de frames a procesar en video
MAX_VIDEO_FRAMES = 1000

# Tamaño máximo de archivo (en MB)
MAX_FILE_SIZE_MB = 50

# Tiempo máximo de procesamiento (en segundos)
MAX_PROCESSING_TIME = 30

# ============================================
# MENSAJES Y TEXTOS
# ============================================
WELCOME_MESSAGE = """
## 👋 Bienvenido al Tutorial Interactivo de OpenCV

Este tutorial te guiará paso a paso en el aprendizaje de procesamiento de imágenes
y visión computacional usando OpenCV.

### 🎯 ¿Cómo empezar?

1. **Selecciona un capítulo** en la barra lateral
2. **Elige un ejercicio** específico
3. **Sigue las instrucciones** interactivas
4. **Experimenta** con los parámetros y observa los resultados

### 📚 Estructura del Curso

- **Capítulo 1:** Fundamentos básicos de OpenCV
- **Capítulo 2:** Procesamiento y transformaciones
- **Capítulo 3:** Detección de características
- **Capítulo 4:** Procesamiento de video
- **Capítulo 5:** Visión computacional avanzada

¡Comienza tu viaje en el mundo de la visión computacional! 🚀
"""

ERROR_MESSAGES = {
    "no_image": "⚠️ Por favor, carga una imagen primero",
    "invalid_image": "❌ La imagen cargada no es válida",
    "processing_error": "❌ Error durante el procesamiento",
    "file_too_large": "❌ El archivo es demasiado grande",
    "unsupported_format": "❌ Formato de archivo no soportado",
}

SUCCESS_MESSAGES = {
    "image_loaded": "✅ Imagen cargada correctamente",
    "processing_complete": "✅ Procesamiento completado",
    "file_saved": "✅ Archivo guardado exitosamente",
}

# ============================================
# CONFIGURACIÓN DE DEMOS
# ============================================
# Parámetros por defecto para diferentes operaciones
DEFAULT_PARAMS = {
    "blur": {
        "gaussian": {"kernel_size": 5, "sigma": 0},
        "median": {"kernel_size": 5},
        "bilateral": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
    },
    "threshold": {
        "simple": {"threshold": 127, "maxval": 255},
        "otsu": {"maxval": 255},
        "adaptive": {"maxval": 255, "blockSize": 11, "C": 2},
    },
    "edge": {
        "canny": {"threshold1": 100, "threshold2": 200},
        "sobel": {"ksize": 3},
        "laplacian": {"ksize": 3},
    },
    "morphology": {
        "kernel_size": 5,
        "iterations": 1,
    },
}

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================
# FEATURES FLAGS (para desarrollo)
# ============================================
ENABLE_DEBUG_MODE = False
ENABLE_PERFORMANCE_METRICS = False
ENABLE_ADVANCED_FEATURES = True
SHOW_CODE_BY_DEFAULT = True