"""
Capítulo 1 - Ejercicio 1: Transformaciones Proyectivas
Aprende a aplicar transformaciones de perspectiva a imágenes
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from core.utils import (
    leer_imagen,
    bgr_to_rgb,
    mostrar_imagen_streamlit,
    comparar_imagenes,
    mostrar_info_imagen
)
from ui.layout import crear_seccion, mostrar_codigo, crear_alerta
from ui.widgets import (
    control_slider,
    panel_control,
    checkbox_simple,
    selector_opciones,
    boton_accion,
    info_tooltip
)


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Transformaciones Proyectivas")
    st.markdown("""
    Las transformaciones proyectivas (o de perspectiva) permiten simular cambios en el ángulo 
    de visualización de una imagen, creando efectos de profundidad y perspectiva.
    """)
    
    st.markdown("---")
    
    # Sidebar para configuración
    with st.sidebar:
        st.markdown("### 🎛️ Configuración de Transformación")
        
        # Selector de imagen
        opcion_imagen = selector_opciones(
            "Selecciona una imagen",
            ["Imagen de ejemplo", "Subir imagen propia"],
            key="img_option"
        )
        
        if opcion_imagen == "Subir imagen propia":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen",
                key="upload_img"
            )
            if archivo:
                from core.utils import cargar_imagen_desde_upload
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("⚠️ Por favor sube una imagen")
                return
        else:
            # Cargar imagen de ejemplo
            img_path = Path("data/images/input.jpg")
            if not img_path.exists():
                # Crear imagen de ejemplo si no existe
                img = crear_imagen_ejemplo()
            else:
                img = leer_imagen(str(img_path))
        
        if img is None:
            st.error("❌ No se pudo cargar la imagen")
            return
    
    # Obtener dimensiones
    rows, cols = img.shape[:2]
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["Transformación Interactiva", "Configuración Manual", "Teoría"])
    
    with tab1:
        transformacion_interactiva(img, rows, cols)
    
    with tab2:
        transformacion_manual(img, rows, cols)
    
    with tab3:
        mostrar_teoria()


def transformacion_interactiva(img, rows, cols):
    """Modo interactivo con presets y controles visuales."""
    
    crear_seccion("Transformación Interactiva", "")
    
    st.markdown("""
    Usa los controles deslizantes para ajustar los puntos de destino y observa 
    cómo cambia la perspectiva de la imagen en tiempo real.
    """)
    
    # Panel de controles
    with panel_control("Ajustes de Perspectiva"):
        
        # Selector de presets
        preset = selector_opciones(
            "Presets de transformación",
            [
                "Personalizado",
                "Inclinación hacia la izquierda",
                "Inclinación hacia la derecha",
                "Efecto trapecio",
                "Perspectiva 3D suave",
                "Rotación en Y"
            ],
            key="preset_select"
        )
        
        # Sliders para ajustar puntos de destino
        st.markdown("**Ajusta los puntos inferiores:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if preset == "Personalizado":
                bottom_left_x = control_slider(
                    "Esquina inferior izquierda (X)",
                    0, cols, int(0.33 * cols),
                    key="bl_x"
                )
            else:
                bottom_left_x = obtener_preset_valor(preset, "bl_x", cols, rows)
                st.info(f"X inferior izq: {bottom_left_x}")
        
        with col2:
            if preset == "Personalizado":
                bottom_right_x = control_slider(
                    "Esquina inferior derecha (X)",
                    0, cols, int(0.66 * cols),
                    key="br_x"
                )
            else:
                bottom_right_x = obtener_preset_valor(preset, "br_x", cols, rows)
                st.info(f"X inferior der: {bottom_right_x}")
        
        # Opciones adicionales
        st.markdown("---")
        mostrar_puntos = checkbox_simple(
            "Mostrar puntos de transformación",
            True,
            key="show_points"
        )
    
    # Aplicar transformación
    src_points = np.float32([
        [0, 0],
        [cols-1, 0],
        [0, rows-1],
        [cols-1, rows-1]
    ])
    
    dst_points = np.float32([
        [0, 0],
        [cols-1, 0],
        [bottom_left_x, rows-1],
        [bottom_right_x, rows-1]
    ])
    
    # Calcular matriz de transformación
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))
    
    # Visualización con puntos y grid si está activado
    img_display = img.copy()
    img_output_display = img_output.copy()
    
    if mostrar_puntos:
        img_display = dibujar_puntos(img_display, src_points, (0, 255, 0))
        img_output_display = dibujar_puntos(img_output_display, dst_points, (0, 0, 255))

    
    # Mostrar resultados
    st.markdown("---")
    crear_seccion("Resultados", "")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img_display, caption="Puntos de origen (verde)")
    
    with col2:
        st.markdown("**Imagen Transformada**")
        mostrar_imagen_streamlit(img_output_display, caption="Puntos de destino (rojo)")
    
    # Información de la transformación
    with st.expander("Matriz de Transformación", expanded=False):
        st.code(f"""
Matriz Proyectiva (3x3):
{projective_matrix}

Puntos de Origen:
{src_points}

Puntos de Destino:
{dst_points}
        """)
    
    # Botón de descarga
    if boton_accion("Guardar imagen transformada", key="save_btn"):
        from core.utils import guardar_imagen
        output_path = Path("data/output/transformed.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if guardar_imagen(img_output, str(output_path)):
            st.success(f"Imagen guardada en: {output_path}")


def transformacion_manual(img, rows, cols):
    """Modo manual para configurar los 4 puntos manualmente."""
    
    crear_seccion("Configuración Manual de Puntos", "")
    
    st.markdown("""
    Especifica manualmente las coordenadas de los 4 puntos de origen y destino 
    para tener control total sobre la transformación.
    """)
    
    with panel_control("Puntos de Origen"):
        col1, col2 = st.columns(2)
        
        with col1:
            src_tl_x = st.number_input("Top-Left X", 0, cols, 0, key="src_tl_x")
            src_tl_y = st.number_input("Top-Left Y", 0, rows, 0, key="src_tl_y")
            src_bl_x = st.number_input("Bottom-Left X", 0, cols, 0, key="src_bl_x")
            src_bl_y = st.number_input("Bottom-Left Y", 0, rows, rows-1, key="src_bl_y")
        
        with col2:
            src_tr_x = st.number_input("Top-Right X", 0, cols, cols-1, key="src_tr_x")
            src_tr_y = st.number_input("Top-Right Y", 0, rows, 0, key="src_tr_y")
            src_br_x = st.number_input("Bottom-Right X", 0, cols, cols-1, key="src_br_x")
            src_br_y = st.number_input("Bottom-Right Y", 0, rows, rows-1, key="src_br_y")
    
    with panel_control("Puntos de Destino"):
        col1, col2 = st.columns(2)
        
        with col1:
            dst_tl_x = st.number_input("Top-Left X", 0, cols, 0, key="dst_tl_x")
            dst_tl_y = st.number_input("Top-Left Y", 0, rows, 0, key="dst_tl_y")
            dst_bl_x = st.number_input("Bottom-Left X", 0, cols, int(0.33*cols), key="dst_bl_x")
            dst_bl_y = st.number_input("Bottom-Left Y", 0, rows, rows-1, key="dst_bl_y")
        
        with col2:
            dst_tr_x = st.number_input("Top-Right X", 0, cols, cols-1, key="dst_tr_x")
            dst_tr_y = st.number_input("Top-Right Y", 0, rows, 0, key="dst_tr_y")
            dst_br_x = st.number_input("Bottom-Right X", 0, cols, int(0.66*cols), key="dst_br_x")
            dst_br_y = st.number_input("Bottom-Right Y", 0, rows, rows-1, key="dst_br_y")
    
    # Crear arrays de puntos
    src_points = np.float32([
        [src_tl_x, src_tl_y],
        [src_tr_x, src_tr_y],
        [src_bl_x, src_bl_y],
        [src_br_x, src_br_y]
    ])
    
    dst_points = np.float32([
        [dst_tl_x, dst_tl_y],
        [dst_tr_x, dst_tr_y],
        [dst_bl_x, dst_bl_y],
        [dst_br_x, dst_br_y]
    ])
    
    # Aplicar transformación
    try:
        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))
        
        # Visualizar
        st.markdown("---")
        comparar_imagenes(img, img_output, ("Original", "Transformada"))
        
    except cv2.error as e:
        st.error(f"❌ Error en la transformación: {str(e)}")
        info_tooltip("Asegúrate de que los puntos formen un cuadrilátero válido.")


def mostrar_teoria():
    """Muestra la teoría y explicación matemática."""
    
    crear_seccion("Teoría de Transformaciones Proyectivas", "")
    
    st.markdown("""
    ### ¿Qué es una Transformación Proyectiva?
    
    Una transformación proyectiva (o de perspectiva) es una función matemática que mapea 
    puntos de un plano a otro plano mediante una matriz 3×3. Esta transformación preserva 
    las líneas rectas pero no necesariamente las distancias o los ángulos.
    
    ### Matriz de Transformación
    
    La matriz proyectiva tiene la forma:
    
    ```
    | a11  a12  a13 |
    | a21  a22  a23 |
    | a31  a32  a33 |
    ```
    
    ### Aplicaciones Prácticas
    
    - **Corrección de perspectiva** en fotografías
    - **Arquitectura** - Vista de edificios
    - **Realidad Aumentada** - Proyección de objetos 3D
    - **Videojuegos** - Renderizado de texturas
    - **Escaneo de documentos** - Corrección de ángulos
    
    ### Fórmula Matemática
    
    Para un punto (x, y) en la imagen original, su posición transformada (x', y') se calcula como:
    
    ```
    x' = (a11*x + a12*y + a13) / (a31*x + a32*y + a33)
    y' = (a21*x + a22*y + a23) / (a31*x + a32*y + a33)
    ```
    
    ### Diferencias con otras Transformaciones
    
    | Transformación | Preserva | Grados de Libertad |
    |----------------|----------|-------------------|
    | Traslación | Todo excepto posición | 2 |
    | Rotación | Forma, tamaño, ángulos | 1 |
    | Afín | Paralelismo | 6 |
    | **Proyectiva** | Líneas rectas | 8 |
    """)
    
    # Código de ejemplo
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo_ejemplo = '''import cv2
import numpy as np

# Leer imagen
img = cv2.imread('imagen.jpg')
rows, cols = img.shape[:2]

# Definir puntos de origen (4 esquinas de la imagen)
src_points = np.float32([
    [0, 0],           # Superior izquierda
    [cols-1, 0],      # Superior derecha
    [0, rows-1],      # Inferior izquierda
    [cols-1, rows-1]  # Inferior derecha
])

# Definir puntos de destino (perspectiva deseada)
dst_points = np.float32([
    [0, 0],
    [cols-1, 0],
    [int(0.33*cols), rows-1],  # Desplazar hacia dentro
    [int(0.66*cols), rows-1]   # Desplazar hacia dentro
])

# Calcular matriz de transformación proyectiva
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Aplicar transformación
img_transformed = cv2.warpPerspective(img, matrix, (cols, rows))

# Mostrar resultado
cv2.imshow('Transformada', img_transformed)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo_ejemplo)
    
    # Tips y mejores prácticas
    st.markdown("---")
    crear_seccion("Tips y Mejores Prácticas", "")
    
    st.markdown("""
    - **Mantén proporciones razonables** - Evita distorsiones extremas
    - **Usa puntos de referencia claros** - Esquinas o bordes bien definidos
    - **Considera el orden de los puntos** - Debe ser consistente (horario o antihorario)
    - **Prueba con diferentes tamaños de salida** - Ajusta según necesites
    - **Cuidado con puntos colineales** - Pueden causar matrices singulares
    """)


def obtener_preset_valor(preset: str, punto: str, cols: int, rows: int) -> int:
    """Obtiene el valor de un preset específico."""
    
    presets = {
        "Inclinación hacia la izquierda": {
            "bl_x": int(0.2 * cols),
            "br_x": int(0.9 * cols)
        },
        "Inclinación hacia la derecha": {
            "bl_x": int(0.1 * cols),
            "br_x": int(0.8 * cols)
        },
        "Efecto trapecio": {
            "bl_x": int(0.25 * cols),
            "br_x": int(0.75 * cols)
        },
        "Perspectiva 3D suave": {
            "bl_x": int(0.15 * cols),
            "br_x": int(0.85 * cols)
        },
        "Rotación en Y": {
            "bl_x": int(0.3 * cols),
            "br_x": int(0.7 * cols)
        }
    }
    
    return presets.get(preset, {"bl_x": int(0.33*cols), "br_x": int(0.66*cols)})[punto]


def dibujar_puntos(img, points, color):
    """Dibuja los puntos de transformación en la imagen."""
    img_copy = img.copy()
    for i, point in enumerate(points):
        cv2.circle(img_copy, tuple(point.astype(int)), 8, color, -1)
        cv2.putText(
            img_copy,
            str(i+1),
            tuple((point + 15).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return img_copy


def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo si no existe ninguna."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Dibujar un patrón de ajedrez
    for i in range(0, 400, 50):
        for j in range(0, 600, 50):
            if (i // 50 + j // 50) % 2 == 0:
                img[i:i+50, j:j+50] = [100, 100, 100]
    
    # Agregar texto
    cv2.putText(
        img,
        "OPENCV",
        (150, 220),
        cv2.FONT_HERSHEY_BOLD,
        2,
        (0, 0, 255),
        3
    )
    
    return img