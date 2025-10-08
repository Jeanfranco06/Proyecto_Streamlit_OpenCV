"""
Capítulo 8 - Ejercicio 8: Detector de Color en Tiempo Real
Aprende a detectar colores específicos usando el espacio de color HSV
y procesamiento de video en tiempo real con la webcam
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from core.utils import (
    leer_imagen,
    # bgr_to_rgb,  # No se usa
    mostrar_imagen_streamlit,
    cargar_imagen_desde_upload
)
from ui.layout import crear_seccion, mostrar_codigo
from ui.widgets import (
    control_slider,
    panel_control,
    checkbox_simple,
    selector_opciones,
    boton_accion,
    info_tooltip,
    entrada_numero
)


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Detector de Color en Tiempo Real")
    st.markdown("""
    Detecta colores específicos en imágenes y video usando el espacio de color **HSV** 
    (Hue, Saturation, Value). Esta técnica es fundamental para visión por computadora, 
    seguimiento de objetos, y sistemas de reconocimiento visual.
    """)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Detector Interactivo",
        "Colores Predefinidos",
        "Análisis HSV",
        "Simulador de Video",
        "Teoría"
    ])
    
    with tab1:
        detector_interactivo()
    
    with tab2:
        colores_predefinidos()
    
    with tab3:
        analisis_hsv()
    
    with tab4:
        simulador_video()
    
    with tab5:
        mostrar_teoria()


def detector_interactivo():
    """Modo interactivo para detectar colores personalizados."""
    
    crear_seccion("Detector de Color Personalizado", "")
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_interactive")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    col_control, col_preview = st.columns([1, 2])
    
    with col_control:
        with panel_control("Rango de Color HSV"):
            
            st.markdown("### Límite Inferior")
            
            lower_h = control_slider(
                "Hue (Matiz) Mínimo",
                0, 179, 60,
                "Tono del color (0-179 en OpenCV)",
                key="lower_h"
            )
            
            lower_s = control_slider(
                "Saturation (Saturación) Mínima",
                0, 255, 100,
                "Intensidad del color",
                key="lower_s"
            )
            
            lower_v = control_slider(
                "Value (Brillo) Mínimo",
                0, 255, 100,
                "Luminosidad del color",
                key="lower_v"
            )
            
            st.markdown("---")
            st.markdown("### Límite Superior")
            
            upper_h = control_slider(
                "Hue (Matiz) Máximo",
                0, 179, 130,
                "Tono del color (0-179 en OpenCV)",
                key="upper_h"
            )
            
            upper_s = control_slider(
                "Saturation (Saturación) Máxima",
                0, 255, 255,
                "Intensidad del color",
                key="upper_s"
            )
            
            upper_v = control_slider(
                "Value (Brillo) Máximo",
                0, 255, 255,
                "Luminosidad del color",
                key="upper_v"
            )
            
            st.markdown("---")
            
            # Filtrado adicional
            aplicar_blur = checkbox_simple(
                "Aplicar filtro Median Blur",
                True,
                "Reduce ruido en la imagen resultante",
                key="apply_blur"
            )
            
            if aplicar_blur:
                ksize = entrada_numero(
                    "Tamaño del kernel",
                    3, 15, 5, 2,
                    key="ksize_blur"
                )
                # Asegurar que sea impar
                if ksize % 2 == 0:
                    ksize += 1
            else:
                ksize = 5
            
            st.markdown("---")
            
            mostrar_mascara = checkbox_simple(
                "Mostrar máscara binaria",
                True,
                key="show_mask"
            )
    
    with col_preview:
        # Crear rangos
        lower = np.array([lower_h, lower_s, lower_v])
        upper = np.array([upper_h, upper_s, upper_v])
        
        # Detectar color
        mask, result = detectar_color(img, lower, upper, aplicar_blur, ksize)
        
        # Mostrar visualización de rango
        crear_seccion("Rango de Color Seleccionado", "")
        visualizar_rango_color(lower, upper)
        
        st.markdown("---")
        crear_seccion("Resultados", "")
        
        if mostrar_mascara:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original**")
                mostrar_imagen_streamlit(img, "", use_column_width=True)
            
            with col2:
                st.markdown("**Máscara**")
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mostrar_imagen_streamlit(mask_rgb, "", use_column_width=True)
            
            with col3:
                st.markdown("**Resultado**")
                mostrar_imagen_streamlit(result, "", use_column_width=True)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original**")
                mostrar_imagen_streamlit(img, "")
            
            with col2:
                st.markdown("**Detección de Color**")
                mostrar_imagen_streamlit(result, "")
        
        
        # Botón de descarga
        if boton_accion("Guardar resultado", key="save_custom"):
            guardar_resultado(result, "color_detection_custom.jpg")


def colores_predefinidos():
    """Presets de colores comunes ya configurados."""
    
    crear_seccion("Colores Predefinidos", "")
    
    st.markdown("""
    Selecciona un color predefinido con rangos HSV optimizados para detección robusta.
    """)
    
    # Definir rangos de colores predefinidos
    colores_presets = {
        "Azul": {
            "lower": [100, 100, 100],
            "upper": [130, 255, 255],
            "descripcion": "Detecta tonos azules (cielo, objetos azules)"
        },
        "Rojo (Parte 1)": {
            "lower": [0, 100, 100],
            "upper": [10, 255, 255],
            "descripcion": "Rojo en el rango bajo del Hue (0-10)"
        },
        "Rojo (Parte 2)": {
            "lower": [170, 100, 100],
            "upper": [179, 255, 255],
            "descripcion": "Rojo en el rango alto del Hue (170-179)"
        },
        "Verde": {
            "lower": [40, 40, 40],
            "upper": [80, 255, 255],
            "descripcion": "Detecta tonos verdes (plantas, césped)"
        },
        "Amarillo": {
            "lower": [20, 100, 100],
            "upper": [30, 255, 255],
            "descripcion": "Tonos amarillos brillantes"
        },
        "Naranja": {
            "lower": [10, 100, 100],
            "upper": [20, 255, 255],
            "descripcion": "Tonos naranjas"
        },
        "Púrpura": {
            "lower": [130, 50, 50],
            "upper": [170, 255, 255],
            "descripcion": "Tonos púrpuras y violetas"
        },
        "Marrón": {
            "lower": [10, 50, 20],
            "upper": [20, 150, 150],
            "descripcion": "Tonos marrones (madera, tierra)"
        },
        "Negro": {
            "lower": [0, 0, 0],
            "upper": [179, 255, 50],
            "descripcion": "Tonos muy oscuros (bajo Value)"
        },
        "Blanco": {
            "lower": [0, 0, 200],
            "upper": [179, 30, 255],
            "descripcion": "Tonos muy claros (bajo Saturation, alto Value)"
        },
        "Cian": {
            "lower": [80, 100, 100],
            "upper": [100, 255, 255],
            "descripcion": "Tonos celestes/cian"
        },
        "Rosa": {
            "lower": [140, 50, 100],
            "upper": [170, 255, 255],
            "descripcion": "Tonos rosados"
        }
    }
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_preset")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Selector de color
    color_seleccionado = selector_opciones(
        "Selecciona un color",
        list(colores_presets.keys()),
        key="preset_color"
    )
    
    preset = colores_presets[color_seleccionado]
    info_tooltip(preset["descripcion"])
    
    st.markdown("---")
    
    # Mostrar rangos HSV del preset
    with st.expander("Ver Rangos HSV", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Límite Inferior:**")
            st.code(f"H: {preset['lower'][0]}\nS: {preset['lower'][1]}\nV: {preset['lower'][2]}")
        
        with col2:
            st.markdown("**Límite Superior:**")
            st.code(f"H: {preset['upper'][0]}\nS: {preset['upper'][1]}\nV: {preset['upper'][2]}")
    
    # Aplicar detección
    lower = np.array(preset["lower"])
    upper = np.array(preset["upper"])
    
    mask, result = detectar_color(img, lower, upper, aplicar_blur=True, ksize=5)
    
    # Visualización
    crear_seccion("Resultado", "")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original**")
        mostrar_imagen_streamlit(img, "", use_column_width=True)
    
    with col2:
        st.markdown("**Máscara**")
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(mask_rgb, "", use_column_width=True)
    
    with col3:
        st.markdown("**Detección**")
        mostrar_imagen_streamlit(result, "", use_column_width=True)
    
    # Estadísticas
    pixels_detectados = np.count_nonzero(mask)
    pixels_totales = mask.size
    porcentaje = (pixels_detectados / pixels_totales) * 100
    
    st.info(f"**Cobertura del color**: {porcentaje:.2f}% de la imagen ({pixels_detectados:,} píxeles)")
    
    # Botón de descarga
    if boton_accion("Guardar resultado", key="save_preset"):
        nombre = color_seleccionado.replace(" ", "_").replace("", "").replace("", "").strip()
        guardar_resultado(result, f"color_detection_{nombre}.jpg")


def analisis_hsv():
    """Análisis detallado del espacio de color HSV."""
    
    crear_seccion("Análisis del Espacio HSV", "")
    
    st.markdown("""
    Explora cómo se distribuyen los colores en el espacio HSV y analiza 
    tu imagen canal por canal.
    """)
    
    # Cargar imagen
    img = cargar_imagen_input("img_source_hsv")
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    # Convertir a HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Visualización de canales
    crear_seccion("Canales HSV Separados", "")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Hue (Matiz)**")
        h_colored = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
        mostrar_imagen_streamlit(h_colored, "", use_column_width=True)
        st.info(f"Rango: 0-179\nMedia: {h.mean():.1f}")
    
    with col2:
        st.markdown("**Saturation (Saturación)**")
        s_gray = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(s_gray, "", use_column_width=True, convertir_rgb=False)
        st.info(f"Rango: 0-255\nMedia: {s.mean():.1f}")
    
    with col3:
        st.markdown("**Value (Brillo)**")
        v_gray = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(v_gray, "", use_column_width=True, convertir_rgb=False)
        st.info(f"Rango: 0-255\nMedia: {v.mean():.1f}")
    
    # Histogramas
    st.markdown("---")
    crear_seccion("Histogramas de Distribución", "")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histograma Hue
    axes[0].hist(h.ravel(), 180, [0, 180], color='red', alpha=0.7)
    axes[0].set_title('Distribución de Hue (Matiz)')
    axes[0].set_xlabel('Valor Hue')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True, alpha=0.3)
    
    # Histograma Saturation
    axes[1].hist(s.ravel(), 256, [0, 256], color='green', alpha=0.7)
    axes[1].set_title('Distribución de Saturation')
    axes[1].set_xlabel('Valor Saturation')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, alpha=0.3)
    
    # Histograma Value
    axes[2].hist(v.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    axes[2].set_title('Distribución de Value')
    axes[2].set_xlabel('Valor Value')
    axes[2].set_ylabel('Frecuencia')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Estadísticas detalladas
    st.markdown("---")
    crear_seccion("Estadísticas Detalladas", "")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Hue")
        st.markdown(f"""
        - **Mínimo**: {h.min()}
        - **Máximo**: {h.max()}
        - **Media**: {h.mean():.2f}
        - **Mediana**: {np.median(h):.2f}
        - **Desv. Std**: {h.std():.2f}
        """)
    
    with col2:
        st.markdown("### Saturation")
        st.markdown(f"""
        - **Mínimo**: {s.min()}
        - **Máximo**: {s.max()}
        - **Media**: {s.mean():.2f}
        - **Mediana**: {np.median(s):.2f}
        - **Desv. Std**: {s.std():.2f}
        """)
    
    with col3:
        st.markdown("### Value")
        st.markdown(f"""
        - **Mínimo**: {v.min()}
        - **Máximo**: {v.max()}
        - **Media**: {v.mean():.2f}
        - **Mediana**: {np.median(v):.2f}
        - **Desv. Std**: {v.std():.2f}
        """)
    


def simulador_video():
    """Simula procesamiento de video con múltiples frames."""
    
    crear_seccion("Simulador de Procesamiento de Video", "")
    
    st.markdown("""
    Simula cómo funcionaría la detección de color en un video en tiempo real.
    En una aplicación real, esto se haría con `cv2.VideoCapture(0)` para capturar 
    desde la webcam.
    """)
    
    # Opciones
    with panel_control("Configuración"):
        color_detectar = selector_opciones(
            "Color a detectar",
            ["Azul", "Rojo", "Verde", "Amarillo", "Personalizado"],
            key="video_color"
        )
        
        if color_detectar == "Personalizado":
            col1, col2 = st.columns(2)
            with col1:
                custom_lower_h = control_slider("Hue Min", 0, 179, 60, key="vid_lh")
                custom_lower_s = control_slider("Sat Min", 0, 255, 100, key="vid_ls")
                custom_lower_v = control_slider("Val Min", 0, 255, 100, key="vid_lv")
            with col2:
                custom_upper_h = control_slider("Hue Max", 0, 179, 130, key="vid_uh")
                custom_upper_s = control_slider("Sat Max", 0, 255, 255, key="vid_us")
                custom_upper_v = control_slider("Val Max", 0, 255, 255, key="vid_uv")
    
    # Definir rangos según selección
    rangos = {
        "Azul": ([100, 100, 100], [130, 255, 255]),
        "Rojo": ([0, 100, 100], [10, 255, 255]),
        "Verde": ([40, 40, 40], [80, 255, 255]),
        "Amarillo": ([20, 100, 100], [30, 255, 255]),
    }
    
    if color_detectar == "Personalizado":
        lower = np.array([custom_lower_h, custom_lower_s, custom_lower_v])
        upper = np.array([custom_upper_h, custom_upper_s, custom_upper_v])
    else:
        lower = np.array(rangos[color_detectar][0])
        upper = np.array(rangos[color_detectar][1])
    
    # Cargar o crear frames de ejemplo
    st.markdown("---")
    crear_seccion("Frame de Ejemplo", "🎬")
    
    img = cargar_imagen_input("img_source_video")
    
    if img is None:
        st.warning("Carga una imagen para simular")
        return
    
    # Procesar frame
    mask, result = detectar_color(img, lower, upper, True, 5)
    
    # Mostrar resultado
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Frame Original**")
        mostrar_imagen_streamlit(img, "", use_column_width=True)
    
    with col2:
        st.markdown("**Máscara**")
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mostrar_imagen_streamlit(mask_rgb, "", use_column_width=True)
    
    with col3:
        st.markdown("**Detección**")
        mostrar_imagen_streamlit(result, "", use_column_width=True)
    
    # Código de ejemplo para video real
    st.markdown("---")
    crear_seccion("Código para Video en Tiempo Real", "")
    
    codigo_video = f'''import cv2
import numpy as np

# Inicializar captura de webcam
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

# Definir rango de color en HSV
lower = np.array({list(lower)})
upper = np.array({list(upper)})

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar para mejor performance
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                      interpolation=cv2.INTER_AREA)
    
    # Convertir a espacio HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear máscara con el rango de color
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    # Aplicar máscara a imagen original
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Aplicar filtro para reducir ruido
    result = cv2.medianBlur(result, ksize=5)
    
    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('Detector de Color', result)
    
    # Salir con tecla ESC (código 27)
    if cv2.waitKey(10) == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
'''
    
    mostrar_codigo(codigo_video)
    
    st.info("""
    **Nota**: Este código captura video en tiempo real desde la webcam.
    - Usa `cv2.VideoCapture(0)` para la webcam predeterminada
    - `cv2.VideoCapture(1)` para webcam externa
    - O proporciona una ruta de archivo de video para procesarlo
    """)


def mostrar_teoria():
    """Explicación teórica de la detección de color."""
    
    crear_seccion("Teoría: Detección de Color con HSV", "")
    
    st.markdown("""
    ### ¿Por qué HSV en lugar de RGB?
    
    El espacio de color **HSV** (Hue, Saturation, Value) es mucho más intuitivo y robusto 
    para detección de colores que RGB por varias razones:
    
    | Característica | RGB | HSV |
    |----------------|-----|-----|
    | **Intuitividad** | Mezcla de canales | Separación clara de color, intensidad y brillo |
    | **Iluminación** | Muy sensible | Robusto a cambios de luz |
    | **Rangos** | Difícil definir | Fácil definir rangos de color |
    | **Sombras** | Cambia todo | Solo afecta Value |
    
    ### Los Tres Componentes de HSV
    
    #### 1. **Hue (Matiz)** - El "color" en sí
    
    - **Rango en OpenCV**: 0-179 (nota: dividido entre 2 del rango estándar 0-360°)
    - **Representa**: El tipo de color (rojo, verde, azul, etc.)
    - **Visualización**: Rueda de colores circular
    
    ```
    Valores aproximados de Hue:
    0-10    → Rojo
    10-20   → Naranja
    20-30   → Amarillo
    30-80   → Verde
    80-100  → Cian
    100-130 → Azul
    130-160 → Púrpura/Violeta
    160-179 → Magenta/Rojo
    ```
    
    **Caso especial del ROJO**: El rojo está en ambos extremos del espectro Hue 
    (0-10 y 170-179), por lo que necesitas dos rangos para detectarlo completamente.
    
    #### 2. **Saturation (Saturación)** - "Pureza" del color
    
    - **Rango**: 0-255
    - **0**: Color grisáceo, desaturado, apagado
    - **255**: Color puro, vibrante, intenso
    
    ```
    Ejemplos:
    Baja saturación (0-50):   Tonos pastel, grises
    Media saturación (50-150): Colores normales
    Alta saturación (150-255): Colores muy vivos
    ```
    
    #### 3. **Value (Brillo)** - Luminosidad
    
    - **Rango**: 0-255
    - **0**: Negro (sin luz)
    - **255**: Máximo brillo
    
    ```
    Ejemplos:
    Bajo Value (0-50):     Muy oscuro, casi negro
    Medio Value (50-150):  Tonos normales
    Alto Value (150-255):  Colores brillantes
    ```
    
    ### Proceso de Detección de Color
    
    El algoritmo sigue estos pasos:
    
    ```
    1. Captura de Frame/Imagen
         ↓
    2. Conversión BGR → HSV
       cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         ↓
    3. Definir Rango de Color
       lower = np.array([h_min, s_min, v_min])
       upper = np.array([h_max, s_max, v_max])
         ↓
    4. Crear Máscara Binaria
       mask = cv2.inRange(hsv_frame, lower, upper)
       - Píxeles en rango → 255 (blanco)
       - Píxeles fuera → 0 (negro)
         ↓
    5. Aplicar Máscara
       result = cv2.bitwise_and(frame, frame, mask=mask)
       - Mantiene píxeles donde mask=255
       - Elimina (negro) donde mask=0
         ↓
    6. Post-procesamiento
       result = cv2.medianBlur(result, ksize=5)
       - Reduce ruido
       - Suaviza resultado
    ```
    
    ### Funciones Clave de OpenCV
    
    #### `cv2.cvtColor()`
    
    Convierte entre espacios de color:
    
    ```python
    # BGR a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # RGB a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # HSV a BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ```
    
    #### `cv2.inRange()`
    
    Crea una máscara binaria basada en rangos:
    
    ```python
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Resultado:
    # - 255 si pixel está dentro del rango
    # - 0 si pixel está fuera del rango
    ```
    
    #### `cv2.bitwise_and()`
    
    Aplica operación AND bit a bit con máscara:
    
    ```python
    result = cv2.bitwise_and(src1, src2, mask=mask)
    
    # Parámetros:
    # - src1, src2: Imágenes de entrada (usualmente la misma)
    # - mask: Máscara binaria (opcional)
    # 
    # Efecto: Mantiene píxeles donde mask=255, 
    #         elimina donde mask=0
    ```
    
    #### `cv2.medianBlur()`
    
    Filtro de suavizado que reduce ruido tipo "sal y pimienta":
    
    ```python
    blurred = cv2.medianBlur(img, ksize=5)
    
    # ksize: Tamaño del kernel (debe ser impar: 3, 5, 7, 9...)
    # - Valores pequeños (3-5): Suavizado leve
    # - Valores grandes (7-15): Suavizado fuerte
    ```
    
    ### Tips para Definir Rangos HSV
    
    #### 1. **Método de Prueba y Error**
    
    - Empieza con rangos amplios
    - Ajusta gradualmente hasta obtener buenos resultados
    - Usa sliders interactivos (como en este ejercicio)
    
    #### 2. **Método del Muestreo**
    
    ```python
    # Selecciona un píxel del color deseado
    pixel_bgr = img[y, x]
    pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Usa esos valores ± tolerancia
    h, s, v = pixel_hsv
    lower = np.array([h-10, s-50, v-50])
    upper = np.array([h+10, s+50, v+50])
    ```
    
    #### 3. **Rangos Conservadores vs Liberales**
    
    | Aspecto | Conservador | Liberal |
    |---------|-------------|---------|
    | **Hue** | Rango estrecho (±5-10) | Rango amplio (±20-30) |
    | **Saturation** | Alto mínimo (100+) | Bajo mínimo (40-70) |
    | **Value** | Alto mínimo (100+) | Bajo mínimo (50-80) |
    | **Resultado** | Menos detecciones, más precisas | Más detecciones, puede haber falsos positivos |
    
    ### Aplicaciones Prácticas
    
    #### 1. **Seguimiento de Objetos**
    
    - Pelotas de colores en deportes
    - Marcadores en realidad aumentada
    - Señales de tráfico
    
    #### 2. **Control de Calidad Industrial**
    
    - Detección de defectos por color
    - Clasificación de productos
    - Verificación de ensamblaje
    
    #### 3. **Interfaces Gestuales**
    
    - Guantes de colores para control
    - Punteros virtuales
    - Reconocimiento de gestos
    
    #### 4. **Análisis de Imágenes Médicas**
    
    - Detección de tejidos específicos
    - Análisis de muestras de sangre
    - Identificación de células
    
    #### 5. **Robótica**
    
    - Navegación por líneas de colores
    - Identificación de objetivos
    - Clasificación de objetos
    
    ### Limitaciones y Desafíos
    
    #### 1. **Iluminación Variable**
    
    Aunque HSV es más robusto que RGB, aún puede verse afectado:
    
    - **Solución**: Normalización de histograma, ajuste automático de rangos
    
    #### 2. **Sombras**
    
    Las sombras reducen el Value, pueden excluir píxeles válidos:
    
    - **Solución**: Ampliar el rango de Value hacia abajo
    
    #### 3. **Reflejo/Brillo Excesivo**
    
    Reduce la saturación, puede perder color:
    
    - **Solución**: Permitir Saturation más baja
    
    #### 4. **Colores Similares**
    
    Algunos colores tienen Hue muy cercano:
    
    - **Solución**: Usar múltiples criterios (forma, tamaño, contexto)
    
    #### 5. **Ruido en la Imagen**
    
    Píxeles aislados causan falsos positivos:
    
    - **Solución**: 
      - Filtros de suavizado (Median, Gaussian Blur)
      - Operaciones morfológicas (erosión, dilatación)
      - Filtrado por área mínima de contornos
    
    ### Mejoras y Técnicas Avanzadas
    
    #### 1. **Operaciones Morfológicas**
    
    ```python
    # Después de crear la máscara
    kernel = np.ones((5,5), np.uint8)
    
    # Erosión: elimina ruido pequeño
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Dilatación: rellena huecos
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apertura: erosión + dilatación (elimina ruido)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Cierre: dilatación + erosión (rellena huecos)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ```
    
    #### 2. **Detección de Contornos**
    
    ```python
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar por área mínima
    min_area = 500
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Dibujar contorno
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Obtener centro
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    ```
    
    #### 3. **Múltiples Rangos de Color**
    
    Para colores como el rojo que están en ambos extremos:
    
    ```python
    # Rojo parte 1
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    # Rojo parte 2
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combinar máscaras
    mask = cv2.bitwise_or(mask1, mask2)
    ```
    
    #### 4. **Ajuste Automático de Histograma**
    
    ```python
    # Ecualización de histograma en canal V
    h, s, v = cv2.split(hsv)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    ```
    
    ### Comparación de Espacios de Color
    
    | Espacio | Ventajas | Desventajas | Uso Ideal |
    |---------|----------|-------------|-----------|
    | **HSV** | Intuitive, robusto a iluminación | Rojo en extremos | Detección de color general |
    | **RGB** | Natural, directo | Sensible a luz | Procesamiento básico |
    | **Lab** | Perceptualmente uniforme | Menos intuitivo | Análisis científico |
    | **YCrCb** | Separa luminancia/crominancia | Complejo | Detección de piel |
    
    ### Código Completo Comentado
    
    """)
    
    codigo_completo = '''import cv2
import numpy as np

def get_frame(cap, scaling_factor):
    """Captura y redimensiona frame de la webcam."""
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, None, 
                          fx=scaling_factor, 
                          fy=scaling_factor,
                          interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':
    # Inicializar captura de video
    cap = cv2.VideoCapture(0)  # 0 = webcam predeterminada
    scaling_factor = 0.5       # Reducir tamaño para mejor performance
    
    # Definir rango de color 'azul' en espacio HSV
    # En OpenCV, Hue va de 0-179 (no 0-360)
    lower = np.array([60, 100, 100])   # [H_min, S_min, V_min]
    upper = np.array([180, 255, 255])  # [H_max, S_max, V_max]
    
    print("Detector de Color Azul Iniciado")
    print("Presiona ESC para salir")
    
    while True:
        # Capturar frame
        frame = get_frame(cap, scaling_factor)
        
        if frame is None:
            print("Error al capturar frame")
            break
        
        # Convertir de BGR (formato de OpenCV) a HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscara: píxeles en rango → 255, fuera → 0
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Aplicar máscara a la imagen original
        # Bitwise AND: mantiene solo píxeles donde mask=255
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Reducir ruido con filtro de mediana
        # ksize=5: kernel de 5x5 (debe ser impar)
        res = cv2.medianBlur(res, ksize=5)
        
        # Opcional: Operaciones morfológicas para limpiar más
        # kernel = np.ones((5,5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Opcional: Encontrar y dibujar contornos
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
        #                                 cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # Mostrar ventanas
        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        
        # También podemos mostrar la máscara
        # cv2.imshow('Mask', mask)
        
        # Esperar tecla
        c = cv2.waitKey(delay=10)  # 10ms = ~100 FPS máx
        
        # Si se presiona ESC (código ASCII 27), salir
        if c == 27:
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Detector cerrado")
'''
    
    mostrar_codigo(codigo_completo)
    
    st.markdown("""
    ### Conceptos Clave para Recordar
    
    1. **HSV es superior a RGB** para detección de color debido a su separación de 
       color (Hue), intensidad (Saturation) y brillo (Value)
    
    2. **`cv2.inRange()`** es la función clave: crea una máscara binaria basada en rangos
    
    3. **Post-procesamiento** es crucial: usa filtros y operaciones morfológicas para 
       limpiar resultados
    
    4. **Rangos amplios** capturan más píxeles pero pueden incluir falsos positivos; 
       **rangos estrechos** son más precisos pero pueden perder píxeles válidos
    
    5. **El rojo es especial**: necesita dos rangos (0-10 y 170-179 en Hue)
    
    6. **Testing iterativo** es la clave: usa sliders/controles para ajustar rangos 
       hasta obtener buenos resultados
    
    ### Referencias y Recursos
    
    - [OpenCV Color Conversions](https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
    - [Understanding HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
    - [Thresholding Tutorial](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
    """)


def detectar_color(img, lower, upper, aplicar_blur=True, ksize=5):
    """
    Detecta un color específico en la imagen.
    
    Args:
        img: Imagen BGR
        lower: Array con límites inferiores HSV [H, S, V]
        upper: Array con límites superiores HSV [H, S, V]
        aplicar_blur: Si aplicar median blur
        ksize: Tamaño del kernel para blur
    
    Returns:
        Tupla (mask, result)
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Crear máscara
    mask = cv2.inRange(hsv, lower, upper)
    
    # Aplicar máscara
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Aplicar blur si se solicita
    if aplicar_blur:
        result = cv2.medianBlur(result, ksize)
    
    return mask, result


def visualizar_rango_color(lower, upper):
    """Visualiza el rango de color seleccionado."""
    
    # Crear imagen del rango de color
    # h_range = np.linspace(lower[0], upper[0], 100).astype(np.uint8)
    # s_range = np.linspace(lower[1], upper[1], 100).astype(np.uint8)
    # v_range = np.linspace(lower[2], upper[2], 100).astype(np.uint8)
    
    # Crear visualización de gradiente
    img_viz = np.zeros((50, 300, 3), dtype=np.uint8)
    
    # Llenar con el color promedio
    h_avg = (lower[0] + upper[0]) // 2
    s_avg = (lower[1] + upper[1]) // 2
    v_avg = (lower[2] + upper[2]) // 2
    
    img_viz[:, :] = [h_avg, s_avg, v_avg]
    img_viz_bgr = cv2.cvtColor(img_viz, cv2.COLOR_HSV2BGR)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Color Central:**")
        mostrar_imagen_streamlit(img_viz_bgr, "", use_column_width=True)
    
    with col2:
        st.markdown(f"""
        **Rango HSV Seleccionado:**
        - Hue: {lower[0]} - {upper[0]}
        - Saturation: {lower[1]} - {upper[1]}
        - Value: {lower[2]} - {upper[2]}
        """)


def cargar_imagen_input(key_base="img_source_colored"):
    """Carga imagen desde archivo o upload."""
    with st.sidebar:
        st.markdown("### Cargar Imagen")
        
        opcion = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key=key_base
        )
        
        if opcion == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen",
                key=f"upload_color_{key_base}"
            )
            if archivo:
                return cargar_imagen_desde_upload(archivo)
            else:
                return None
        else:
            img_path = Path("data/images/detectar_color.jpg")
            if img_path.exists():
                return leer_imagen(str(img_path))
            else:
                # Crear imagen de ejemplo con colores
                return crear_imagen_colores_ejemplo()


def crear_imagen_colores_ejemplo():
    """Crea una imagen de ejemplo con múltiples colores."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Crear bloques de colores en HSV
    colores_hsv = [
        ([0, 255, 255], "Rojo"),      # Rojo
        ([30, 255, 255], "Amarillo"),  # Amarillo
        ([60, 255, 255], "Verde"),     # Verde
        ([90, 255, 255], "Cian"),      # Cian
        ([120, 255, 255], "Azul"),     # Azul
        ([150, 255, 255], "Magenta"),  # Magenta
    ]
    
    block_width = 600 // len(colores_hsv)
    
    for i, (color_hsv, nombre) in enumerate(colores_hsv):
        # Crear bloque de color
        x_start = i * block_width
        x_end = (i + 1) * block_width
        
        color_block = np.zeros((400, block_width, 3), dtype=np.uint8)
        color_block[:, :] = color_hsv
        
        # Convertir a BGR
        color_bgr = cv2.cvtColor(color_block, cv2.COLOR_HSV2BGR)
        
        img[:, x_start:x_end] = color_bgr
        
        # Añadir texto
        cv2.putText(img, nombre, (x_start + 10, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img


def guardar_resultado(img, nombre):
    """Guarda la imagen resultante."""
    from core.utils import guardar_imagen
    output_path = Path("data/output") / nombre
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if guardar_imagen(img, str(output_path)):
        st.success(f"Imagen guardada en: {output_path}")
    else:
        st.error("Error al guardar la imagen")