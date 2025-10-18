"""
Capítulo 6 - Ejercicio 6: Seam Carving - Eliminación de Objetos (VERSIÓN CORREGIDA)
"""
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
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
    entrada_numero
)


def run():
    """Función principal del ejercicio."""
    
    # Header del ejercicio
    st.title("Seam Carving - Eliminación de Objetos")
    st.markdown("""
    Elimina objetos de tus imágenes de forma inteligente usando **Seam Carving**, 
    una técnica que preserva el contenido importante mientras remueve elementos no deseados.
    ¡Haz que los objetos desaparezcan como si nunca hubieran existido!
    """)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "Eliminación Interactiva",
        "Redimensionado Inteligente",
        "Proceso Técnico",
        "Teoría"
    ])
    
    with tab1:
        eliminacion_interactiva()
    
    with tab2:
        redimensionado_inteligente()
    
    with tab3:
        proceso_tecnico()
    
    with tab4:
        mostrar_teoria()


def asegurar_uint8(img):
    """Asegura que la imagen esté en formato uint8 con valores válidos."""
    if img.dtype != np.uint8:
        # Normalizar a 0-255
        if img.max() > 255:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def mostrar_imagen_segura(img, caption=""):
    """Muestra imagen con validaciones de seguridad."""
    try:
        # Asegurar que sea uint8
        img = asegurar_uint8(img)
        
        # Validar dimensiones
        if len(img.shape) not in [2, 3]:
            st.error(f"Formato de imagen inválido: {img.shape}")
            return
        
        # Convertir BGR a RGB si es color
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = bgr_to_rgb(img)
        
        st.image(img, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error mostrando imagen: {e}")


def eliminacion_interactiva():
    """Interfaz interactiva para eliminar objetos."""
    
    crear_seccion("Eliminación Interactiva de Objetos", "")
    
    st.markdown("""
    Dibuja un rectángulo sobre el objeto que deseas eliminar y observa cómo 
    desaparece de la imagen preservando el contenido circundante.
    """)
    
    # Sidebar para controles
    with st.sidebar:
        st.markdown("### Configuración")
        
        # Selector de imagen
        opcion_imagen = selector_opciones(
            "Fuente de imagen",
            ["Imagen con silla", "Imagen con persona", "Subir imagen"],
            key="img_source_seam"
        )
        
        if opcion_imagen == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube una imagen",
                key="upload_seam"
            )
            if archivo:
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("Por favor sube una imagen")
                return
        else:
            # Usar imágenes de ejemplo
            img_paths = {
                "Imagen con silla": "data/images/chair_scene.jpg",
                "Imagen con persona": "data/images/person_scene.jpg"
            }
            img_path = Path(img_paths.get(opcion_imagen, img_paths["Imagen con silla"]))
            
            if img_path.exists():
                img = leer_imagen(str(img_path))
            else:
                st.warning(f"No se encontró {img_path}. Usa 'Subir imagen' para probar.")
                img = crear_imagen_demo()
        
        if img is None:
            st.error("No se pudo cargar la imagen")
            return
        
        # Asegurar que sea uint8
        img = asegurar_uint8(img)
        
        st.markdown("---")
        st.markdown("### Parámetros")
        
        extra_seams = control_slider(
            "Seams extra a remover",
            0, 50, 10,
            "Cuántos seams adicionales remover además del ancho del objeto",
            key="extra_seams"
        )
        
        energy_method = selector_opciones(
            "Método de energía",
            ["Gradiente", "Laplaciano", "Sobel"],
            key="energy_method"
        )
        
        protection_size = control_slider(
            "Tamaño de protección",
            0, 20, 5,
            "Píxeles adicionales alrededor del objeto a proteger",
            key="protection_size"
        )
        
        mostrar_proceso = checkbox_simple(
            "Mostrar proceso paso a paso",
            False,
            "Ver cada seam siendo removido (más lento)",
            key="show_process"
        )
    
    # Interfaz de selección de región
    st.markdown("### Selecciona el Objeto a Eliminar")
    
    # SOLUCIÓN: No usar streamlit-drawable-canvas, usar entrada manual confiable
    st.markdown("**Ingresa las coordenadas del rectángulo a eliminar:**")
    st.markdown("*(x, y) = esquina superior izquierda | (ancho, alto) = dimensiones*")
    
    # Mostrar imagen con referencia
    st.markdown("**Imagen de referencia:**")
    mostrar_imagen_segura(img, f"Tamaño: {img.shape[1]} x {img.shape[0]}px")
    
    col_x, col_y, col_w, col_h = st.columns(4)
    
    with col_x:
        rect_x = st.number_input(
            "X (columna inicio)",
            min_value=0,
            max_value=max(0, img.shape[1] - 1),
            value=min(100, img.shape[1] // 4),
            step=1,
            key="rect_x_input"
        )
    
    with col_y:
        rect_y = st.number_input(
            "Y (fila inicio)",
            min_value=0,
            max_value=max(0, img.shape[0] - 1),
            value=min(100, img.shape[0] // 4),
            step=1,
            key="rect_y_input"
        )
    
    with col_w:
        rect_w = st.number_input(
            "Ancho",
            min_value=10,
            max_value=img.shape[1],
            value=min(100, img.shape[1] // 3),
            step=1,
            key="rect_w_input"
        )
    
    with col_h:
        rect_h = st.number_input(
            "Alto",
            min_value=10,
            max_value=img.shape[0],
            value=min(100, img.shape[0] // 3),
            step=1,
            key="rect_h_input"
        )
    
    # Vista previa del rectángulo
    st.markdown("---")
    st.markdown("**Vista previa de la selección:**")
    
    img_preview = img.copy()
    
    # Validar y ajustar coordenadas
    rect_x = max(0, min(int(rect_x), img.shape[1] - 1))
    rect_y = max(0, min(int(rect_y), img.shape[0] - 1))
    rect_w = max(10, min(int(rect_w), img.shape[1] - rect_x))
    rect_h = max(10, min(int(rect_h), img.shape[0] - rect_y))
    
    # Dibujar rectángulo en la vista previa
    cv2.rectangle(img_preview, (rect_x, rect_y), 
                 (rect_x + rect_w, rect_y + rect_h), 
                 (0, 255, 0), 3)
    
    # Agregar texto con info del rectángulo
    cv2.putText(img_preview, f"({rect_x}, {rect_y}) - {rect_w}x{rect_h}",
               (max(10, rect_x - 5), max(20, rect_y - 5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    mostrar_imagen_segura(img_preview, "Rectángulo en verde")
    
    st.info(f"""
    **Región seleccionada:**
    - Posición: ({rect_x}, {rect_y})
    - Dimensiones: {rect_w} x {rect_h}px
    - Área: {rect_w * rect_h} píxeles
    """)
    
    st.markdown("---")
    
    # Botón de procesamiento
    if boton_accion("Eliminar Objeto", key="remove_btn_final"):
        procesar_eliminacion(
            img,
            (rect_x, rect_y, rect_w, rect_h),
            extra_seams,
            energy_method,
            protection_size,
            mostrar_proceso
        )


def redimensionado_inteligente():
    """Redimensionado de imagen usando seam carving."""
    
    crear_seccion("Redimensionado Inteligente", "")
    
    st.markdown("""
    A diferencia del redimensionado tradicional que distorsiona la imagen, 
    el **seam carving** preserva el contenido importante mientras cambia las dimensiones.
    """)
    
    # Sidebar para controles
    with st.sidebar:
        st.markdown("### Configuración")
        
        opcion_imagen = selector_opciones(
            "Fuente de imagen",
            ["Paisaje", "Edificio", "Subir imagen"],
            key="img_source_resize"
        )
        
        if opcion_imagen == "Subir imagen":
            from ui.widgets import subir_archivo
            archivo = subir_archivo(
                ["png", "jpg", "jpeg"],
                "Sube una imagen",
                key="upload_resize"
            )
            if archivo:
                img = cargar_imagen_desde_upload(archivo)
            else:
                st.warning("Por favor sube una imagen")
                return
        else:
            img_paths = {
                "Paisaje": "data/images/landscape.jpg",
                "Edificio": "data/images/building.jpg"
            }
            img_path = Path(img_paths.get(opcion_imagen, img_paths["Paisaje"]))
            
            if img_path.exists():
                img = leer_imagen(str(img_path))
            else:
                img = crear_imagen_demo()
        
        if img is None:
            st.error("No se pudo cargar la imagen")
            return
        
        img = asegurar_uint8(img)
        
        st.markdown("---")
        st.markdown("### Dimensiones")
        
        original_width = img.shape[1]
        original_height = img.shape[0]
        
        st.info(f"Tamaño original: {original_width} x {original_height}px")
        
        nuevo_ancho = control_slider(
            "Nuevo ancho (%)",
            50, 100, 80,
            key="new_width_percent"
        )
        
        nuevo_ancho_px = int(original_width * nuevo_ancho / 100)
        seams_to_remove = original_width - nuevo_ancho_px
        
        st.metric("Seams a remover", seams_to_remove)
        
        st.markdown("---")
        
        energy_method_resize = selector_opciones(
            "Método de energía",
            ["Gradiente", "Laplaciano", "Sobel"],
            key="energy_method_resize"
        )
    
    # Mostrar imagen original
    st.markdown("### Imagen Original")
    mostrar_imagen_segura(img, f"Tamaño: {original_width} x {original_height}px")
    
    st.markdown("---")
    
    # Botón de procesamiento
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if boton_accion("Redimensionar con Seam Carving", key="resize_seam_btn"):
            procesar_redimensionado(img, seams_to_remove, energy_method_resize)
    
    st.markdown("---")
    
    # Comparación con redimensionado tradicional
    st.markdown("### Comparación de Métodos")
    
    if boton_accion("Ver comparación con resize tradicional", key="compare_btn"):
        comparar_metodos_redimensionado(img, nuevo_ancho_px)


def proceso_tecnico():
    """Visualización del proceso técnico de seam carving."""
    
    crear_seccion("Proceso Técnico: Seam Carving", "")
    
    st.markdown("""
    Veamos paso a paso cómo funciona el algoritmo de Seam Carving para eliminar 
    contenido de una imagen de forma inteligente.
    """)
    
    # Cargar imagen de ejemplo
    img = crear_imagen_demo_simple()
    img = asegurar_uint8(img)
    
    st.markdown("### Paso 1: Imagen Original")
    mostrar_imagen_segura(img, f"Tamaño: {img.shape[1]} x {img.shape[0]}px")
    
    st.markdown("---")
    
    # Paso 2: Calcular energía
    st.markdown("### Paso 2: Calcular Mapa de Energía")
    
    st.markdown("""
    El mapa de energía identifica las regiones importantes de la imagen usando gradientes.
    Los píxeles con alta energía (cambios bruscos) son importantes y deben preservarse.
    """)
    
    energy = compute_energy_matrix(img)
    
    # Normalizar para visualización
    energy_vis = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    energy_colored = cv2.applyColorMap(energy_vis, cv2.COLORMAP_JET)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_segura(img, "")
    with col2:
        st.markdown("**Mapa de Energía**")
        mostrar_imagen_segura(energy_colored, "Rojo=Alta energía, Azul=Baja energía")
    
    st.markdown("---")
    
    # Paso 3: Encontrar seam
    st.markdown("### Paso 3: Encontrar Seam de Mínima Energía")
    
    st.markdown("""
    Un **seam** es un camino conectado de píxeles desde el tope hasta el fondo de la imagen.
    Buscamos el seam con la menor energía total usando **programación dinámica**.
    """)
    
    seam = find_vertical_seam(img, energy)
    
    # Visualizar seam
    img_seam = img.copy()
    for i, j in enumerate(seam):
        cv2.circle(img_seam, (j, i), 2, (0, 255, 0), -1)
    
    mostrar_imagen_segura(img_seam, "Seam de mínima energía (verde)")
    
    st.markdown("---")
    
    # Paso 4: Remover seam
    st.markdown("### Paso 4: Remover Seam")
    
    img_removed = remove_vertical_seam(img.copy(), seam)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Antes ({img.shape[1]}px ancho)**")
        mostrar_imagen_segura(img, "")
    with col2:
        st.markdown(f"**Después ({img_removed.shape[1]}px ancho)**")
        mostrar_imagen_segura(img_removed, "")
    
    st.markdown("---")
    
    # Paso 5: Proceso completo
    st.markdown("### Paso 5: Remover Múltiples Seams")
    
    num_seams_demo = st.slider("Número de seams a remover:", 1, 50, 20, key="num_seams_demo")
    
    if st.button("▶ Procesar", key="process_demo_btn"):
        with st.spinner("Removiendo seams..."):
            img_result = img.copy()
            energy_current = compute_energy_matrix(img_result)
            
            progress_bar = st.progress(0)
            
            for i in range(num_seams_demo):
                seam = find_vertical_seam(img_result, energy_current)
                img_result = remove_vertical_seam(img_result, seam)
                energy_current = compute_energy_matrix(img_result)
                progress_bar.progress((i + 1) / num_seams_demo)
            
            st.success(f"{num_seams_demo} seams removidos exitosamente!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original ({img.shape[1]}px)**")
                mostrar_imagen_segura(img, "")
            with col2:
                st.markdown(f"**Resultado ({img_result.shape[1]}px)**")
                mostrar_imagen_segura(img_result, "")
            
            reduction = ((img.shape[1] - img_result.shape[1]) / img.shape[1]) * 100
            st.metric("Reducción de ancho", f"{reduction:.1f}%")


def mostrar_teoria():
    """Sección teórica sobre seam carving."""
    
    crear_seccion("Teoría: Seam Carving", "")
    
    st.markdown("""
    ### ¿Qué es Seam Carving?
    
    **Seam Carving** (también conocido como *content-aware resizing*) es una técnica 
    desarrollada por **Shai Avidan y Ariel Shamir en 2007** que permite redimensionar 
    imágenes preservando el contenido visualmente importante.
    
    [Resto del contenido igual...]
    """)


# ============================================================================
# FUNCIONES DE PROCESAMIENTO (CORREGIDAS)
# ============================================================================

def compute_energy_matrix(img, method="Gradiente"):
    """Calcula la matriz de energía de la imagen."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if method == "Gradiente":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
    
    elif method == "Laplaciano":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        energy = np.abs(laplacian)
    
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        energy = np.sqrt(sobelx**2 + sobely**2)
    
    return energy.astype(np.float64)


def find_vertical_seam(img, energy):
    """Encuentra el seam vertical de mínima energía usando programación dinámica."""
    rows, cols = energy.shape
    M = energy.copy()
    
    # Forward pass
    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                M[i, j] += min(M[i-1, j], M[i-1, j+1])
            elif j == cols - 1:
                M[i, j] += min(M[i-1, j-1], M[i-1, j])
            else:
                M[i, j] += min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
    
    # Backtracking
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    
    for i in range(rows - 2, -1, -1):
        j = seam[i + 1]
        
        if j == 0:
            candidates = M[i, j:j+2]
            seam[i] = j + np.argmin(candidates)
        elif j == cols - 1:
            candidates = M[i, j-1:j+1]
            seam[i] = j - 1 + np.argmin(candidates)
        else:
            candidates = M[i, j-1:j+2]
            seam[i] = j - 1 + np.argmin(candidates)
    
    return seam


def remove_vertical_seam(img, seam):
    """Remueve un seam vertical de la imagen."""
    rows, cols = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    if channels == 1:
        output = np.zeros((rows, cols - 1), dtype=img.dtype)
    else:
        output = np.zeros((rows, cols - 1, channels), dtype=img.dtype)
    
    for i in range(rows):
        j = seam[i]
        if channels == 1:
            output[i, :j] = img[i, :j]
            output[i, j:] = img[i, j+1:]
        else:
            output[i, :j] = img[i, :j]
            output[i, j:] = img[i, j+1:]
    
    return output


def add_vertical_seam(img, seam, offset=0):
    """Agrega un seam vertical duplicándolo."""
    rows, cols = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    if channels == 1:
        output = np.zeros((rows, cols + 1), dtype=img.dtype)
    else:
        output = np.zeros((rows, cols + 1, channels), dtype=img.dtype)
    
    for i in range(rows):
        j = seam[i]
        
        if channels == 1:
            output[i, :j] = img[i, :j]
            if j < cols - 1:
                output[i, j] = (img[i, j].astype(np.float32) + img[i, j+1].astype(np.float32)) / 2
            else:
                output[i, j] = img[i, j]
            output[i, j+1:] = img[i, j:]
        else:
            output[i, :j] = img[i, :j]
            if j < cols - 1:
                output[i, j] = (img[i, j].astype(np.float32) + img[i, j+1].astype(np.float32)) / 2
            else:
                output[i, j] = img[i, j]
            output[i, j+1:] = img[i, j:]
    
    return asegurar_uint8(output)


def procesar_eliminacion(img, rect, extra_seams, energy_method, protection_size, mostrar_proceso):
    """Procesa la eliminación de un objeto de la imagen."""
    
    x, y, w, h = rect
    
    # Validar rectángulo
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        st.error("El rectángulo está fuera de los límites de la imagen")
        return
    
    if w <= 0 or h <= 0:
        st.error("El rectángulo debe tener dimensiones positivas")
        return
    
    num_seams = w + extra_seams
    
    st.info(f"Procesando: Se removerán {num_seams} seams (puede tomar varios minutos)...")
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fase 1: Remover objeto
        status_text.text("Fase 1/2: Removiendo objeto...")
        
        img_temp = img.copy()
        
        for i in range(num_seams):
            energy = compute_energy_matrix(img_temp, energy_method)
            
            x_protected = max(0, x - protection_size)
            y_protected = max(0, y - protection_size)
            w_protected = min(img_temp.shape[1] - x_protected, w + 2 * protection_size - i)
            h_protected = min(img_temp.shape[0] - y_protected, h + 2 * protection_size)
            
            if w_protected > 0 and h_protected > 0:
                energy[y_protected:y_protected+h_protected, 
                       x_protected:x_protected+w_protected] = 0
            
            seam = find_vertical_seam(img_temp, energy)
            img_temp = remove_vertical_seam(img_temp, seam)
            
            progress = (i + 1) / (2 * num_seams)
            progress_bar.progress(progress)
            status_text.text(f"Fase 1/2: Seam {i+1}/{num_seams} removido")
            
            if mostrar_proceso and i % 5 == 0:
                mostrar_imagen_segura(img_temp, f"Seams removidos: {i+1}")
        
        # Fase 2: Rellenar
        status_text.text("Fase 2/2: Rellenando región...")
        
        img_output = img_temp.copy()
        
        for i in range(num_seams):
            energy = compute_energy_matrix(img_temp, energy_method)
            seam = find_vertical_seam(img_temp, energy)
            img_temp = remove_vertical_seam(img_temp, seam)
            img_output = add_vertical_seam(img_output, seam, i)
            
            progress = 0.5 + (i + 1) / (2 * num_seams)
            progress_bar.progress(progress)
            status_text.text(f"Fase 2/2: Seam {i+1}/{num_seams} agregado")
            
            if mostrar_proceso and i % 5 == 0:
                mostrar_imagen_segura(img_output, f"Seams agregados: {i+1}")
        
        progress_bar.progress(1.0)
        status_text.text("Procesamiento completado!")
    
    st.success(f"¡Objeto eliminado exitosamente! Se procesaron {num_seams * 2} seams en total.")
    
    # Mostrar resultados
    st.markdown("---")
    st.markdown("### Resultado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_segura(img, f"Tamaño: {img.shape[1]} x {img.shape[0]}px")
    
    with col2:
        st.markdown("**Objeto Eliminado**")
        mostrar_imagen_segura(img_output, f"Tamaño: {img_output.shape[1]} x {img_output.shape[0]}px")
    
    # Botón de descarga
    if boton_accion("Guardar resultado", key="save_removed"):
        guardar_resultado(img_output, "object_removed.jpg")


def procesar_redimensionado(img, num_seams, energy_method):
    """Procesa el redimensionado de imagen usando seam carving."""
    
    if num_seams <= 0:
        st.warning("No hay seams para remover")
        return
    
    st.info(f"Redimensionando imagen: removiendo {num_seams} seams...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    img_result = img.copy()
    
    for i in range(num_seams):
        energy = compute_energy_matrix(img_result, energy_method)
        seam = find_vertical_seam(img_result, energy)
        img_result = remove_vertical_seam(img_result, seam)
        
        progress = (i + 1) / num_seams
        progress_bar.progress(progress)
        status_text.text(f"Seam {i+1}/{num_seams} removido")
    
    progress_bar.progress(1.0)
    status_text.text("Redimensionado completado!")
    
    st.success(f"Imagen redimensionada de {img.shape[1]}px a {img_result.shape[1]}px")
    
    # Mostrar resultados
    st.markdown("---")
    st.markdown("### Resultado del Redimensionado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Original ({img.shape[1]}px ancho)**")
        mostrar_imagen_segura(img, "")
    
    with col2:
        st.markdown(f"**Redimensionado ({img_result.shape[1]}px ancho)**")
        mostrar_imagen_segura(img_result, "")
    
    reduction = ((img.shape[1] - img_result.shape[1]) / img.shape[1]) * 100
    st.metric("Reducción de ancho", f"{reduction:.1f}%")
    
    if boton_accion("Guardar resultado", key="save_resized"):
        guardar_resultado(img_result, "seam_carved.jpg")


def comparar_metodos_redimensionado(img, nuevo_ancho):
    """Compara seam carving con redimensionado tradicional."""
    
    st.markdown("### Comparación: Seam Carving vs Resize Tradicional")
    
    with st.spinner("Procesando comparación..."):
        # Método 1: Resize tradicional
        img_resized = cv2.resize(img, (nuevo_ancho, img.shape[0]), 
                                interpolation=cv2.INTER_AREA)
        
        # Método 2: Seam carving
        num_seams = img.shape[1] - nuevo_ancho
        img_seam = img.copy()
        
        progress_bar = st.progress(0)
        
        for i in range(num_seams):
            energy = compute_energy_matrix(img_seam)
            seam = find_vertical_seam(img_seam, energy)
            img_seam = remove_vertical_seam(img_seam, seam)
            progress_bar.progress((i + 1) / num_seams)
    
    # Mostrar comparación
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Original ({img.shape[1]}px)**")
        mostrar_imagen_segura(img, "")
    
    with col2:
        st.markdown(f"**Resize Tradicional ({nuevo_ancho}px)**")
        mostrar_imagen_segura(img_resized, "")
        st.caption("Distorsiona el contenido uniformemente")
    
    with col3:
        st.markdown(f"**Seam Carving ({nuevo_ancho}px)**")
        mostrar_imagen_segura(img_seam, "")
        st.caption("Preserva contenido importante")
    
    st.info("""
    **Observa la diferencia:**
    - **Resize tradicional**: Comprime toda la imagen uniformemente, distorsionando objetos
    - **Seam carving**: Remueve regiones de baja importancia, preservando objetos importantes
    """)


def crear_imagen_demo():
    """Crea una imagen de demostración con objetos para eliminar."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Fondo con textura
    for i in range(0, 600, 20):
        cv2.line(img, (i, 0), (i, 400), (220, 220, 220), 1)
    for i in range(0, 400, 20):
        cv2.line(img, (0, i), (600, i), (220, 220, 220), 1)
    
    # Objeto principal
    cv2.rectangle(img, (200, 150), (350, 300), (100, 100, 255), -1)
    cv2.rectangle(img, (200, 150), (350, 300), (50, 50, 200), 3)
    cv2.putText(img, "OBJETO", (230, 235), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 255, 255), 2)
    
    # Objetos de fondo
    cv2.circle(img, (100, 100), 40, (100, 255, 100), -1)
    cv2.circle(img, (500, 300), 50, (100, 255, 100), -1)
    
    return img


def crear_imagen_demo_simple():
    """Crea una imagen simple para demostración del proceso técnico."""
    img = np.ones((200, 300, 3), dtype=np.uint8) * 180
    
    cv2.rectangle(img, (50, 50), (100, 150), (100, 100, 255), -1)
    cv2.circle(img, (200, 100), 40, (100, 255, 100), -1)
    cv2.line(img, (150, 0), (150, 200), (255, 100, 100), 3)
    
    return img


def guardar_resultado(img, filename):
    """Guarda la imagen resultado."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), img)
    
    st.success(f"Imagen guardada en: {output_path}")
    
    # Ofrecer descarga
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Descargar imagen",
            data=file,
            file_name=filename,
            mime="image/jpeg"
        )


if __name__ == "__main__":
    run()