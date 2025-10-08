"""
Capítulo 2 - Ejercicio 2: Filtro Vignette (Viñeta)
Aprende a crear efectos de viñeta profesionales usando kernels Gaussianos
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
    cargar_imagen_desde_upload
)
from ui.layout import crear_seccion, mostrar_codigo, crear_alerta
from ui.widgets import (
    control_slider,
    panel_control,
    checkbox_simple,
    selector_opciones,
    boton_accion,
    info_tooltip,
    selector_color,
    entrada_numero
)


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Filtro Vignette (Viñeta)")
    st.markdown("""
    El efecto vignette oscurece los bordes de una imagen para enfocar la atención en el centro.
    Es ampliamente usado en fotografía profesional y cinematografía para crear atmósfera dramática.
    """)
    
    st.markdown("---")
    
    # Cargar imagen
    img = cargar_imagen_input()
    
    if img is None:
        st.warning("Por favor carga una imagen para continuar")
        return
    
    rows, cols = img.shape[:2]
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Filtro Interactivo",
        "Estilos Preconfigurados",
        "Análisis Técnico",
        "Teoría"
    ])
    
    with tab1:
        filtro_interactivo(img, rows, cols)
    
    with tab2:
        estilos_preconfigurados(img, rows, cols)
    
    with tab3:
        analisis_tecnico(img, rows, cols)
    
    with tab4:
        mostrar_teoria()


def filtro_interactivo(img, rows, cols):
    """Modo interactivo con controles ajustables."""
    
    crear_seccion("Controles de Viñeta", "")
    
    col_control, col_preview = st.columns([1, 2])
    
    with col_control:
        with panel_control("Parámetros del Filtro"):
            
            # Intensidad del efecto
            intensidad = control_slider(
                "Intensidad",
                0, 100, 70,
                "Controla qué tan oscuros serán los bordes",
                key="intensidad"
            )
            
            # Tamaño del área central
            sigma_x = entrada_numero(
                "Dispersión Horizontal (Sigma X)",
                50.0, 500.0, 200.0, 10.0,
                ayuda="Controla el tamaño horizontal del área clara",
                key="sigma_x"
            )
            
            sigma_y = entrada_numero(
                "Dispersión Vertical (Sigma Y)",
                50.0, 500.0, 200.0, 10.0,
                ayuda="Controla el tamaño vertical del área clara",
                key="sigma_y"
            )
            
            st.markdown("---")
            
            # Opciones avanzadas
            usar_mismo_sigma = checkbox_simple(
                "Usar mismo valor para ambos ejes",
                False,
                "Mantiene la viñeta circular en lugar de elíptica",
                key="mismo_sigma"
            )
            
            if usar_mismo_sigma:
                sigma_y = sigma_x
            
            st.markdown("---")
            
            # Color de viñeta
            tipo_vignette = selector_opciones(
                "Tipo de Viñeta",
                ["Oscura (clásica)", "Clara (invertida)", "Color personalizado"],
                key="tipo_vignette"
            )
            
            color_vignette = None
            if tipo_vignette == "Color personalizado":
                color_hex = selector_color(
                    "Color de los bordes",
                    "#000000",
                    key="color_vignette"
                )
                # Convertir hex a BGR
                color_vignette = hex_to_bgr(color_hex)
            
            st.markdown("---")
            
            # Opciones de visualización
            mostrar_mascara = checkbox_simple(
                "Mostrar máscara de viñeta",
                False,
                key="show_mask"
            )
            
            mostrar_comparacion = checkbox_simple(
                "Vista comparativa",
                True,
                key="show_comparison"
            )
    
    with col_preview:
        # Generar viñeta
        output, mask = aplicar_vignette(
            img, rows, cols, 
            sigma_x, sigma_y, 
            intensidad,
            tipo_vignette,
            color_vignette
        )
        
        if mostrar_mascara:
            crear_seccion("Máscara de Viñeta", "")
            # Mostrar la máscara en formato visual
            mask_visual = (mask * 255).astype(np.uint8)
            if len(mask_visual.shape) == 2:
                mask_visual = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR)
            mostrar_imagen_streamlit(mask_visual, "Máscara Gaussiana", convertir_rgb=False)
            st.markdown("---")
        
        crear_seccion("Resultado", "")
        
        if mostrar_comparacion:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                mostrar_imagen_streamlit(img, "")
            with col2:
                st.markdown("**Con Viñeta**")
                mostrar_imagen_streamlit(output, "")
        else:
            mostrar_imagen_streamlit(output, "Imagen con Viñeta")
        
        # Botón de descarga
        if boton_accion("💾 Guardar imagen", key="save_interactive"):
            guardar_resultado(output, "vignette_custom.jpg")


def estilos_preconfigurados(img, rows, cols):
    """Presets de estilos de viñeta populares."""
    
    crear_seccion("Estilos Predefinidos", "")
    
    st.markdown("""
    Selecciona un estilo preconfigurado inspirado en estilos fotográficos populares.
    """)
    
    # Definir presets
    presets = {
        "Clásico Suave": {"sigma_x": 250, "sigma_y": 250, "intensidad": 60, "tipo": "Oscura (clásica)"},
        "Drama Intenso": {"sigma_x": 150, "sigma_y": 150, "intensidad": 85, "tipo": "Oscura (clásica)"},
        "Retrato Elegante": {"sigma_x": 200, "sigma_y": 180, "intensidad": 70, "tipo": "Oscura (clásica)"},
        "Cinematográfico": {"sigma_x": 180, "sigma_y": 220, "intensidad": 75, "tipo": "Oscura (clásica)"},
        "Vintage": {"sigma_x": 220, "sigma_y": 220, "intensidad": 55, "tipo": "Oscura (clásica)"},
        "Brillo Central": {"sigma_x": 200, "sigma_y": 200, "intensidad": 80, "tipo": "Clara (invertida)"},
    }
    
    # Selector de preset
    preset_seleccionado = selector_opciones(
        "Selecciona un estilo",
        list(presets.keys()),
        key="preset_style"
    )
    
    params = presets[preset_seleccionado]
    
    # Mostrar descripción del preset
    descripciones = {
        "Clásico Suave": "Viñeta suave y natural, perfecta para retratos casuales",
        "Drama Intenso": "Efecto dramático con bordes muy oscuros, ideal para fotografía artística",
        "Retrato Elegante": "Viñeta elíptica que enfatiza el rostro en retratos",
        "Cinematográfico": "Estilo de película con formato panorámico",
        "Vintage": "Efecto nostálgico sutil inspirado en fotografía analógica",
        "Brillo Central": "Ilumina el centro en lugar de oscurecer los bordes",
    }
    
    info_tooltip(descripciones[preset_seleccionado])
    
    st.markdown("---")
    
    # Aplicar preset
    output, _ = aplicar_vignette(
        img, rows, cols,
        params["sigma_x"],
        params["sigma_y"],
        params["intensidad"],
        params["tipo"]
    )
    
    # Mostrar galería de estilos
    crear_seccion("Vista Previa", "")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown(f"**{preset_seleccionado}**")
        mostrar_imagen_streamlit(output, "")
    
    # Botón de descarga
    if boton_accion("Guardar estilo", key="save_preset"):
        guardar_resultado(output, f"vignette_{preset_seleccionado.lower().replace(' ', '_')}.jpg")


def analisis_tecnico(img, rows, cols):
    """Visualización técnica del filtro Gaussiano."""
    
    crear_seccion("Análisis Técnico del Filtro", "")
    
    st.markdown("""
    Explora cómo funcionan los kernels Gaussianos y su impacto en la máscara de viñeta.
    """)
    
    with panel_control("Parámetros de Análisis"):
        sigma_analisis = control_slider(
            "Sigma del Kernel",
            50, 400, 200,
            key="sigma_analisis"
        )
    
    # Generar kernels
    kernel_x = cv2.getGaussianKernel(cols, sigma_analisis)
    kernel_y = cv2.getGaussianKernel(rows, sigma_analisis)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    
    # Visualizaciones
    tab_a, tab_b, tab_c = st.tabs(["Kernels 1D", "Máscara 2D", "Perfil de Intensidad"])
    
    with tab_a:
        st.markdown("### Kernels Gaussianos 1D")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Kernel Horizontal (X)**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(kernel_x, 'b-', linewidth=2)
            ax.set_title("Kernel Gaussiano X")
            ax.set_xlabel("Posición")
            ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.info(f"Tamaño: {len(kernel_x)} elementos")
        
        with col2:
            st.markdown("**Kernel Vertical (Y)**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(kernel_y, 'r-', linewidth=2)
            ax.set_title("Kernel Gaussiano Y")
            ax.set_xlabel("Posición")
            ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.info(f"Tamaño: {len(kernel_y)} elementos")
    
    with tab_b:
        st.markdown("### Máscara 2D Resultante")
        
        # Visualizar la máscara como mapa de calor
        mask_visual = (mask).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_visual, cv2.COLORMAP_JET)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Máscara en Escala de Grises**")
            mostrar_imagen_streamlit(
                cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2BGR),
                "",
                convertir_rgb=False
            )
        
        with col2:
            st.markdown("**Mapa de Calor**")
            mostrar_imagen_streamlit(mask_colored, "")
        
        st.info(f"Rango de valores: [{mask.min():.2f}, {mask.max():.2f}]")
    
    with tab_c:
        st.markdown("### Perfil de Intensidad")
        
        # Perfil horizontal (fila del medio)
        mid_row = rows // 2
        profile_h = mask[mid_row, :]
        
        # Perfil vertical (columna del medio)
        mid_col = cols // 2
        profile_v = mask[:, mid_col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Perfil Horizontal (Centro)**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(profile_h, 'b-', linewidth=2)
            ax.set_title("Intensidad a lo largo del eje X")
            ax.set_xlabel("Posición X")
            ax.set_ylabel("Intensidad")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=255/2, color='r', linestyle='--', alpha=0.5, label='50% intensidad')
            ax.legend()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Perfil Vertical (Centro)**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(profile_v, 'r-', linewidth=2)
            ax.set_title("Intensidad a lo largo del eje Y")
            ax.set_xlabel("Posición Y")
            ax.set_ylabel("Intensidad")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=255/2, color='b', linestyle='--', alpha=0.5, label='50% intensidad')
            ax.legend()
            st.pyplot(fig)
            plt.close()
    


def mostrar_teoria():
    """Explicación teórica del filtro vignette."""
    
    crear_seccion("Teoría: Filtro Vignette", "")
    
    st.markdown("""
    ### ¿Qué es el Efecto Vignette?
    
    El efecto **vignette** (viñeta) es una reducción de brillo o saturación en la periferia 
    de una imagen comparada con el centro. Originalmente era un defecto óptico en lentes 
    antiguos, pero ahora se usa intencionalmente como técnica artística.
    
    ### Propósito y Aplicaciones
    
    - **Dirección de la atención**: Guía la mirada del espectador hacia el centro
    - **Atmósfera dramática**: Añade profundidad y emoción a retratos
    - **Estilo vintage**: Simula el look de fotografía analógica antigua
    - **Cinematografía**: Usado extensivamente en películas para crear mood
    - **Fotografía de producto**: Elimina distracciones en los bordes
    
    ### Cómo Funciona Técnicamente
    
    El filtro vignette se crea multiplicando la imagen por una **máscara Gaussiana 2D**:
    
    1. **Generar kernels Gaussianos 1D** para X e Y:
       ```python
       kernel_x = cv2.getGaussianKernel(cols, sigma_x)
       kernel_y = cv2.getGaussianKernel(rows, sigma_y)
       ```
    
    2. **Crear máscara 2D** mediante producto externo:
       ```python
       kernel = kernel_y * kernel_x.T
       ```
    
    3. **Normalizar** la máscara:
       ```python
       mask = 255 * kernel / np.linalg.norm(kernel)
       ```
    
    4. **Aplicar a cada canal** de color:
       ```python
       for i in range(3):
           output[:,:,i] = output[:,:,i] * mask
       ```
    
    ### La Función Gaussiana
    
    La distribución Gaussiana tiene la forma:
    
    ```
    G(x) = (1 / (σ√(2π))) * e^(-(x-μ)²/(2σ²))
    ```
    
    Donde:
    - **σ (sigma)**: Desviación estándar - controla la "dispersión" (qué tan amplio es el área clara)
    - **μ (mu)**: Media - ubicación del centro (usualmente el centro de la imagen)
    
    ### Variaciones del Efecto
    
    | Tipo | Descripción | Uso |
    |------|-------------|-----|
    | **Oscura** | Bordes oscuros (clásico) | Retratos, drama |
    | **Clara** | Centro brillante | Efectos de ensueño |
    | **Elíptica** | Forma ovalada | Retratos verticales |
    | **Circular** | Forma redonda | Fotos cuadradas |
    | **Colorizada** | Bordes con color | Efectos creativos |
    
    ### Parámetros Clave
    
    - **Sigma (σ)**: 
      - Valores bajos (50-150): Viñeta pronunciada, área clara pequeña
      - Valores altos (200-400): Viñeta suave, transición gradual
    
    - **Intensidad**:
      - 0-30%: Sutil, apenas perceptible
      - 40-70%: Moderado, equilibrado
      - 70-100%: Dramático, bordes muy oscuros
    
    ### Tips Profesionales
    
    - **Mantén sutileza**: Un vignette demasiado fuerte se ve artificial
    - **Considera la composición**: El sujeto debe estar en el centro
    - **Ajusta según el contenido**: Retratos vs paisajes necesitan diferentes intensidades
    - **Usa viñetas elípticas** para retratos verticales
    - **Evita en fotos con información en los bordes**
    """)
    
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo = '''import cv2
import numpy as np

# Leer imagen
img = cv2.imread('imagen.jpg')
rows, cols = img.shape[:2]

# Generar kernels Gaussianos
kernel_x = cv2.getGaussianKernel(cols, 200)
kernel_y = cv2.getGaussianKernel(rows, 200)

# Crear máscara 2D mediante producto externo
kernel = kernel_y * kernel_x.T

# Normalizar la máscara
mask = 255 * kernel / np.linalg.norm(kernel)

# Crear copia de salida
output = np.copy(img)

# Aplicar la máscara a cada canal RGB/BGR
for i in range(3):
    output[:,:,i] = output[:,:,i] * mask

# Mostrar resultado
cv2.imshow('Vignette', output)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo)
    
    st.markdown("---")
    crear_seccion("Comparación: Antes vs Después", "")
    
    st.markdown("""
    ### Efectos Visuales del Vignette
    
    - **Antes**: Imagen plana, atención distribuida uniformemente
    - **Después**: Enfoque central, profundidad perceptual, atmósfera mejorada
    
    ### Diferencias con Otros Filtros
    
    | Filtro | Efecto | Complejidad |
    |--------|--------|-------------|
    | Vignette | Oscurece bordes gradualmente | Baja |
    | Viñeta radial | Patrón circular perfecto | Media |
    | Desenfoque selectivo | Desenfoca bordes | Alta |
    | Gradiente lineal | Transición en una dirección | Baja |
    """)


def aplicar_vignette(img, rows, cols, sigma_x, sigma_y, intensidad, tipo="Oscura (clásica)", color=None):
    """
    Aplica el efecto vignette a una imagen.
    
    Args:
        img: Imagen de entrada
        rows, cols: Dimensiones
        sigma_x, sigma_y: Parámetros Gaussianos
        intensidad: Intensidad del efecto (0-100)
        tipo: Tipo de viñeta
        color: Color personalizado (BGR) o None
    
    Returns:
        Tupla (imagen_procesada, mascara)
    """
    # Generar kernels Gaussianos
    kernel_x = cv2.getGaussianKernel(cols, int(sigma_x))
    kernel_y = cv2.getGaussianKernel(rows, int(sigma_y))
    
    # Crear máscara 2D
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.linalg.norm(kernel)
    
    # Ajustar intensidad
    intensidad_factor = intensidad / 100.0
    
    if tipo == "Clara (invertida)":
        # Invertir la máscara
        mask = 1.0 - (mask * intensidad_factor)
        mask = np.clip(mask, 0, 1)
    else:
        # Viñeta oscura normal
        mask = 1.0 - ((1.0 - mask) * intensidad_factor)
        mask = np.clip(mask, 0, 1)
    
    output = np.copy(img).astype(np.float32)
    
    if color is not None and tipo == "Color personalizado":
        # Aplicar color personalizado
        for i in range(3):
            output[:,:,i] = output[:,:,i] * mask + color[i] * (1 - mask)
    else:
        # Aplicar máscara normal
        for i in range(3):
            output[:,:,i] = output[:,:,i] * mask
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output, mask


def cargar_imagen_input():
    """Carga imagen desde archivo o upload."""
    with st.sidebar:
        st.markdown("### Cargar Imagen")
        
        opcion = selector_opciones(
            "Fuente de imagen",
            ["Imagen de ejemplo", "Subir imagen"],
            key="img_source"
        )
        
        if opcion == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube tu imagen",
                key="upload_vignette"
            )
            if archivo:
                return cargar_imagen_desde_upload(archivo)
            else:
                return None
        else:
            img_path = Path("data/images/input.jpg")
            if img_path.exists():
                return leer_imagen(str(img_path))
            else:
                # Crear imagen de ejemplo
                return crear_imagen_ejemplo()


def crear_imagen_ejemplo():
    """Crea una imagen de ejemplo colorida."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Crear gradiente de colores
    for i in range(400):
        for j in range(600):
            img[i, j] = [
                int(255 * j / 600),  # B
                int(255 * i / 400),  # G
                128                   # R
            ]
    
    # Añadir texto
    cv2.putText(
        img,
        "VIGNETTE FILTER",
        (120, 200),
        cv2.FONT_HERSHEY_BOLD,
        1.5,
        (255, 255, 255),
        3
    )
    
    return img


def hex_to_bgr(hex_color):
    """Convierte color hex a BGR."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR


def guardar_resultado(img, nombre):
    """Guarda la imagen resultante."""
    from core.utils import guardar_imagen
    output_path = Path("data/output") / nombre
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if guardar_imagen(img, str(output_path)):
        st.success(f"Imagen guardada en: {output_path}")
    else:
        st.error("Error al guardar la imagen")