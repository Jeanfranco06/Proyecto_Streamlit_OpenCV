"""
Capítulo 6 - Ejercicio 6: Seam Carving - Eliminación de Objetos
Aprende a eliminar objetos de imágenes usando la técnica de Seam Carving
preservando el contenido importante de la escena.
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
                # Crear imagen de demostración
                img = crear_imagen_demo()
        
        if img is None:
            return
        
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
    
    # Usar columnas para el layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Dibuja un rectángulo sobre el objeto:**")
        
        # Usar streamlit-drawable-canvas si está disponible
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Convertir imagen para canvas
            img_rgb = bgr_to_rgb(img)
            pil_img = Image.fromarray(img_rgb)
            
            # Canvas interactivo
            background_url = pil_to_data_url(pil_img)
            st.write("Tipo de imagen:", type(pil_img))
            st.write("Modo:", getattr(pil_img, "mode", "sin modo"))
            st.write("Tamaño:", getattr(pil_img, "size", "sin tamaño"))
            st.image(pil_img, caption="Verificación previa", use_container_width=True)
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=pil_img,
                update_streamlit=True,
                height=img.shape[0],
                width=img.shape[1],
                drawing_mode="rect",
                key="canvas_seam",
            )
            
            has_drawable = True
            
        except ImportError:
            has_drawable = False
            st.info("Instala `streamlit-drawable-canvas` para selección interactiva: `pip install streamlit-drawable-canvas`")
            
            # Alternativa: Entrada manual de coordenadas
            st.markdown("**Ingresa las coordenadas manualmente:**")
            
            col_x, col_y, col_w, col_h = st.columns(4)
            with col_x:
                rect_x = entrada_numero("X", 0, img.shape[1], 100, 1, key="rect_x")
            with col_y:
                rect_y = entrada_numero("Y", 0, img.shape[0], 100, 1, key="rect_y")
            with col_w:
                rect_w = entrada_numero("Ancho", 10, img.shape[1], 100, 1, key="rect_w")
            with col_h:
                rect_h = entrada_numero("Alto", 10, img.shape[0], 100, 1, key="rect_h")
            
            # Dibujar rectángulo de vista previa
            img_preview = img.copy()
            cv2.rectangle(img_preview, (rect_x, rect_y), 
                         (rect_x + rect_w, rect_y + rect_h), 
                         (0, 255, 0), 2)
            mostrar_imagen_streamlit(img_preview, "Vista previa de la selección")
            
            canvas_result = None
    
    with col2:
        st.markdown("**Instrucciones:**")
        st.info("""
        1. Dibuja un rectángulo alrededor del objeto que deseas eliminar
        2. Ajusta los parámetros si es necesario
        3. Haz clic en 'Eliminar Objeto'
        4. Espera el procesamiento (puede tomar tiempo)
        """)
        
        st.warning("**Importante:** El procesamiento puede tomar varios minutos dependiendo del tamaño de la región.")
    
    st.markdown("---")
    
    # Botón de procesamiento
    if has_drawable and canvas_result is not None and canvas_result.json_data is not None:
        # Extraer rectángulo del canvas
        objects = canvas_result.json_data.get("objects", [])
        
        if len(objects) > 0:
            rect = objects[-1]  # Último rectángulo dibujado
            
            # Manejar diferentes versiones de streamlit-drawable-canvas
            try:
                # Intentar acceder como diccionario
                rect_x = int(rect.get("left", 0))
                rect_y = int(rect.get("top", 0))
                rect_w = int(rect.get("width", 0))
                rect_h = int(rect.get("height", 0))
            except Exception:
                # Si falla, intentar acceder como lista/tupla de valores
                try:
                    rect_x = int(rect[0])
                    rect_y = int(rect[1])
                    rect_w = int(rect[2])
                    rect_h = int(rect[3])
                except Exception:
                    st.error("❌ Error al extraer coordenadas del rectángulo. Usa entrada manual.")
                    rect_x = rect_y = rect_w = rect_h = 0
            
            if rect_w > 0 and rect_h > 0:
                st.success(f"Región seleccionada: x={rect_x}, y={rect_y}, w={rect_w}, h={rect_h}")
                
                if boton_accion("Eliminar Objeto", key="remove_btn"):
                    procesar_eliminacion(
                        img, 
                        (rect_x, rect_y, rect_w, rect_h),
                        extra_seams,
                        energy_method,
                        protection_size,
                        mostrar_proceso
                    )
            else:
                st.warning("El rectángulo debe tener dimensiones válidas")
        else:
            st.info("Dibuja un rectángulo sobre el objeto a eliminar")
    
    elif not has_drawable:
        # Usar coordenadas manuales
        if boton_accion("Eliminar Objeto", key="remove_btn_manual"):
            procesar_eliminacion(
                img,
                (rect_x, rect_y, rect_w, rect_h),
                extra_seams,
                energy_method,
                protection_size,
                mostrar_proceso
            )


def pil_to_data_url(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

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
        
        # Selector de imagen
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
            return
        
        st.markdown("---")
        st.markdown("### Dimensiones")
        
        original_width = img.shape[1]
        original_height = img.shape[0]
        
        st.info(f"Tamaño original: {original_width} x {original_height}px")
        
        # Control de ancho
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
    mostrar_imagen_streamlit(img, f"Tamaño: {original_width} x {original_height}px")
    
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
    
    st.markdown("### Paso 1: Imagen Original")
    mostrar_imagen_streamlit(img, f"Tamaño: {img.shape[1]} x {img.shape[0]}px")
    
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
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown("**Mapa de Energía**")
        mostrar_imagen_streamlit(energy_colored, "Rojo=Alta energía, Azul=Baja energía")
    
    st.code("""
# Cálculo de energía usando Sobel
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
energy = np.abs(grad_x) + np.abs(grad_y)
    """)
    
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
    
    mostrar_imagen_streamlit(img_seam, "Seam de mínima energía (verde)")
    
    st.code("""
# Programación dinámica para encontrar seam óptimo
M = energy.copy()
for i in range(1, M.shape[0]):
    for j in range(M.shape[1]):
        # Considerar 3 posibles padres
        left = M[i-1, max(j-1, 0)]
        mid = M[i-1, j]
        right = M[i-1, min(j+1, M.shape[1]-1)]
        M[i, j] += min(left, mid, right)
    """)
    
    st.markdown("---")
    
    # Paso 4: Remover seam
    st.markdown("### Paso 4: Remover Seam")
    
    img_removed = remove_vertical_seam(img.copy(), seam)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Antes ({img.shape[1]}px ancho)**")
        mostrar_imagen_streamlit(img, "")
    with col2:
        st.markdown(f"**Después ({img_removed.shape[1]}px ancho)**")
        mostrar_imagen_streamlit(img_removed, "")
    
    st.code("""
# Remover píxeles del seam
img_removed = np.zeros((img.shape[0], img.shape[1]-1, 3), dtype=img.dtype)
for i, j in enumerate(seam):
    # Copiar píxeles antes y después del seam
    img_removed[i, :j] = img[i, :j]
    img_removed[i, j:] = img[i, j+1:]
    """)
    
    st.markdown("---")
    
    # Paso 5: Proceso completo
    st.markdown("### Paso 5: Remover Múltiples Seams")
    
    num_seams_demo = st.slider("Número de seams a remover:", 1, 50, 20, key="num_seams_demo")
    
    if st.button("▶Procesar", key="process_demo_btn"):
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
                mostrar_imagen_streamlit(img, "")
            with col2:
                st.markdown(f"**Resultado ({img_result.shape[1]}px)**")
                mostrar_imagen_streamlit(img_result, "")
            
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
    
    ### Conceptos Clave
    
    #### **1. Seam (Costura)**
    
    Un seam es un camino conectado de píxeles de baja energía que atraviesa la imagen:
    
    - **Seam vertical**: De arriba hacia abajo
    - **Seam horizontal**: De izquierda a derecha
    - **Conectividad**: Cada píxel se conecta con su vecino (8-conectividad)
    
    ```
    Ejemplo de seam vertical:
    
    [0,0] [0,1] [0,2] [0,3] [0,4]
      ↓     ↓     
    [1,0] [1,1]●[1,2] [1,3] [1,4]
            ↓
    [2,0] [2,1] [2,2]●[2,3] [2,4]
                  ↓
    [3,0] [3,1] [3,2] [3,3]●[3,4]
    ```
    
    #### **2. Energía (Importancia)**
    
    La energía mide la "importancia" visual de cada píxel:
    
    **Métodos comunes:**
    
    - **Gradiente**: `E = |∂I/∂x| + |∂I/∂y|`
    - **Sobel**: Derivadas aproximadas con kernel Sobel
    - **Laplaciano**: `E = ∇²I`
    - **Saliency**: Basado en atención visual
    
    **Alta energía** → Bordes, esquinas, detalles importantes
    **Baja energía** → Regiones uniformes, cielo, fondos
    
    ### Algoritmo de Seam Carving
    
    #### **Fase 1: Reducción de Tamaño**
    
    ```python
    # Pseudocódigo
    while image_width > target_width:
        energy = compute_energy(image)
        seam = find_min_energy_seam(energy)
        image = remove_seam(image, seam)
    ```
    
    #### **Fase 2: Búsqueda de Seam (Programación Dinámica)**
    
    ```
    1. Inicializar: M[0, j] = energy[0, j] para todo j
    
    2. Para cada fila i de 1 a height-1:
         Para cada columna j de 0 a width-1:
             M[i, j] = energy[i, j] + min(
                 M[i-1, j-1],  # diagonal izquierda
                 M[i-1, j],     # arriba
                 M[i-1, j+1]    # diagonal derecha
             )
    
    3. Encontrar j_min = argmin(M[height-1, :])
    
    4. Backtracking desde (height-1, j_min) hasta (0, :)
    ```
    
    **Complejidad**: O(width × height)
    
    ### Eliminación de Objetos
    
    Para eliminar un objeto de la imagen:
    
    **Estrategia:**
    
    1. **Marcar región**: Usuario selecciona área del objeto
    2. **Forzar energía baja**: Establecer energía = 0 en la región
    3. **Remover seams**: Los seams pasarán por el objeto (baja energía)
    4. **Rellenar**: Agregar seams para mantener tamaño original
    
    ```python
    def remove_object(img, rect):
        x, y, w, h = rect
        num_seams = w + extra_padding
        
        # Fase 1: Remover objeto
        for i in range(num_seams):
            energy = compute_energy(img)
            energy[y:y+h, x:x+w] = 0  # Forzar baja energía
            seam = find_seam(energy)
            img = remove_seam(img, seam)
        
        # Fase 2: Rellenar para mantener tamaño
        for i in range(num_seams):
            seam = find_seam(compute_energy(img))
            img = add_seam(img, seam)  # Duplicar seam
        
        return img
    ```
    
    ### Seam Vertical vs Horizontal
    
    | Aspecto | Vertical | Horizontal |
    |---------|----------|------------|
    | **Dirección** | Arriba → Abajo | Izquierda → Derecha |
    | **Reduce** | Ancho | Alto |
    | **Conectividad** | (i, j) → (i+1, j±1) | (i, j) → (i±1, j+1) |
    | **Uso común** | Más frecuente | Menos frecuente |
    
    ### Ventajas del Seam Carving
    
    **Preserva contenido importante** - No distorsiona objetos
    **Content-aware** - Inteligente, no uniforme
    **Elimina regiones repetitivas** - Cielo, agua, fondos
    **Versátil** - Reducir, ampliar, eliminar objetos
    **No requiere segmentación** - Funciona automáticamente
    
    ### Limitaciones
    
    **Lento** - Procesamiento intensivo para imágenes grandes
    **Objetos delgados** - Pueden distorsionarse (personas, postes)
    **Patrones repetitivos** - Puede crear discontinuidades
    **Imágenes uniformes** - Menos efectivo sin estructura clara
    **Grandes cambios** - Reducir >30% puede causar artefactos
    
    ### Variaciones y Mejoras
    
    #### **Forward Energy (Energía Hacia Adelante)**
    
    En lugar de usar energía actual, predice la energía después de remover el seam:
    
    ```python
    # Energía tradicional
    M[i,j] = energy[i,j] + min(M[i-1, j-1], M[i-1,j], M[i-1,j+1])
    
    # Forward energy
    M[i,j] = min(
        M[i-1,j-1] + cost_left(i,j),
        M[i-1,j]   + cost_mid(i,j),
        M[i-1,j+1] + cost_right(i,j)
    )
    ```
    
    **Ventaja**: Reduce artefactos visuales
    
    #### **Multi-size Images**
    
    Pre-computar varios tamaños y guardar información de seams:
    
    - Permite cambio de tamaño en tiempo real
    - Usado en responsive design
    
    #### **Seam Protection**
    
    Marcar regiones que NO deben removerse:
    
    ```python
    # Aumentar energía en regiones protegidas
    energy[protected_mask] = np.inf
    ```
    
    ### Aplicaciones Reales
    
    - **Fotografía móvil** - Redimensionado inteligente para pantallas
    - **Retoque fotográfico** - Eliminar elementos no deseados
    - **Post-producción** - Ajustar aspect ratio de videos
    - **Diseño editorial** - Adaptar imágenes a layouts
    - **Arte digital** - Manipulación creativa de imágenes
    - **Televisión** - Convertir 4:3 a 16:9
    
    ### Comparación con Otras Técnicas
    
    | Técnica | Calidad | Velocidad | Preserva Contenido |
    |---------|---------|-----------|-------------------|
    | **Crop** | ⭐⭐⭐ | ⚡⚡⚡⚡⚡ | ⭐⭐ (pierde info) |
    | **Resize/Scale** | ⭐⭐ | ⚡⚡⚡⚡⚡ | ⭐ (distorsiona) |
    | **Seam Carving** | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐⭐ |
    | **Content-Fill (AI)** | ⭐⭐⭐⭐⭐ | ⚡ | ⭐⭐⭐⭐ |
    
    ### Tips para Mejores Resultados
    
    **Imágenes con fondos uniformes** - Mejor candidato
    **Objetos bien definidos** - Bordes claros
    **Cambios moderados** - Reducir <30% del tamaño
    **Proteger regiones importantes** - Aumentar su energía
    **Usar forward energy** - Menos artefactos
    **Pre-procesamiento** - Aumentar contraste puede ayudar
    
    ### Paper Original
    
    **"Seam Carving for Content-Aware Image Resizing"**
    Shai Avidan & Ariel Shamir, SIGGRAPH 2007
    
    *Paper altamente influyente que introdujo el concepto al mundo*
    
    ### Matemática Detrás del Algoritmo
    
    #### **Función de Energía (Ejemplo: Gradiente)**
    
    ```
    E(I) = |∂I/∂x| + |∂I/∂y|
    
    Donde:
    - I(x,y) = Intensidad en (x,y)
    - ∂I/∂x = Derivada parcial en x
    - ∂I/∂y = Derivada parcial en y
    ```
    
    #### **Programación Dinámica**
    
    ```
    Objetivo: Minimizar Σ E(si)
    
    M[i,j] = energía acumulada mínima hasta (i,j)
    
    Recurrencia:
    M[i,j] = E[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
    
    Casos base:
    M[0,j] = E[0,j] para todo j
    ```
    
    ### Forward Energy (Mejora Importante)
    
    En lugar de minimizar energía existente, minimiza energía **insertada**:
    
    ```
    CL(i,j) = |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j-1)|
    CU(i,j) = |I(i,j+1) - I(i,j-1)|
    CR(i,j) = |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j+1)|
    
    M[i,j] = min(
        M[i-1,j-1] + CL(i,j),
        M[i-1,j]   + CU(i,j),
        M[i-1,j+1] + CR(i,j)
    )
    ```
    """)
    
    st.markdown("---")
    crear_seccion("Código de Ejemplo", "")
    
    codigo = '''import cv2
import numpy as np

def compute_energy_matrix(img):
    """Calcula la matriz de energía usando gradientes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular gradientes
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Energía = magnitud del gradiente
    energy = np.abs(grad_x) + np.abs(grad_y)
    
    return energy

def find_vertical_seam(img, energy):
    """Encuentra el seam vertical de mínima energía."""
    rows, cols = energy.shape
    M = energy.copy()
    
    # Programación dinámica
    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                M[i,j] += min(M[i-1,j], M[i-1,j+1])
            elif j == cols-1:
                M[i,j] += min(M[i-1,j-1], M[i-1,j])
            else:
                M[i,j] += min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
    
    # Backtracking
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(M[-1])
    
    for i in range(rows-2, -1, -1):
        j = seam[i+1]
        if j == 0:
            seam[i] = j + np.argmin(M[i, j:j+2])
        elif j == cols-1:
            seam[i] = j-1 + np.argmin(M[i, j-1:j+1])
        else:
            seam[i] = j-1 + np.argmin(M[i, j-1:j+2])
    
    return seam

def remove_vertical_seam(img, seam):
    """Remueve un seam vertical de la imagen."""
    rows, cols = img.shape[:2]
    output = np.zeros((rows, cols-1, 3), dtype=img.dtype)
    
    for i in range(rows):
        j = seam[i]
        output[i, :j] = img[i, :j]
        output[i, j:] = img[i, j+1:]
    
    return output

def remove_object(img, rect):
    """Elimina un objeto de la imagen."""
    x, y, w, h = rect
    num_seams = w + 10  # Seams extra
    
    # Fase 1: Remover
    for i in range(num_seams):
        energy = compute_energy_matrix(img)
        # Forzar seam a pasar por el objeto
        energy[y:y+h, x:x+w-i] = 0
        seam = find_vertical_seam(img, energy)
        img = remove_vertical_seam(img, seam)
        print(f'Seams removidos: {i+1}/{num_seams}')
    
    # Fase 2: Rellenar
    img_output = img.copy()
    for i in range(num_seams):
        energy = compute_energy_matrix(img)
        seam = find_vertical_seam(img, energy)
        img = remove_vertical_seam(img, seam)
        img_output = add_vertical_seam(img_output, seam)
        print(f'Seams agregados: {i+1}/{num_seams}')
    
    return img_output

# Uso
img = cv2.imread('image.jpg')
rect = (100, 100, 150, 200)  # x, y, w, h
result = remove_object(img, rect)

cv2.imshow('Original', img)
cv2.imshow('Resultado', result)
cv2.waitKey(0)
'''
    
    mostrar_codigo(codigo, "python")


# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def compute_energy_matrix(img, method="Gradiente"):
    """Calcula la matriz de energía de la imagen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    
    return energy


def find_vertical_seam(img, energy):
    """Encuentra el seam vertical de mínima energía usando programación dinámica."""
    rows, cols = energy.shape
    M = energy.copy()
    
    # Forward pass - Programación dinámica
    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                M[i, j] += min(M[i-1, j], M[i-1, j+1])
            elif j == cols - 1:
                M[i, j] += min(M[i-1, j-1], M[i-1, j])
            else:
                M[i, j] += min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
    
    # Backtracking - Encontrar el camino óptimo
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
            # Promediar píxeles vecinos para el nuevo píxel
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
    
    return output.astype(img.dtype)


def procesar_eliminacion(img, rect, extra_seams, energy_method, protection_size, mostrar_proceso):
    """Procesa la eliminación de un objeto de la imagen."""
    
    x, y, w, h = rect
    
    # Validar rectángulo
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        st.error("❌ El rectángulo está fuera de los límites de la imagen")
        return
    
    if w <= 0 or h <= 0:
        st.error("❌ El rectángulo debe tener dimensiones positivas")
        return
    
    num_seams = w + extra_seams
    
    st.info(f"Procesando: Se removerán {num_seams} seams (puede tomar varios minutos)...")
    
    # Crear contenedor para el progreso
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fase 1: Remover objeto
        status_text.text("Fase 1/2: Removiendo objeto...")
        
        img_temp = img.copy()
        
        for i in range(num_seams):
            # Calcular energía
            energy = compute_energy_matrix(img_temp, energy_method)
            
            # Ajustar región con protección
            x_protected = max(0, x - protection_size)
            y_protected = max(0, y - protection_size)
            w_protected = min(img_temp.shape[1] - x_protected, w + 2 * protection_size - i)
            h_protected = min(img_temp.shape[0] - y_protected, h + 2 * protection_size)
            
            # Forzar seam a pasar por el objeto (energía = 0)
            if w_protected > 0 and h_protected > 0:
                energy[y_protected:y_protected+h_protected, 
                       x_protected:x_protected+w_protected] = 0
            
            # Encontrar y remover seam
            seam = find_vertical_seam(img_temp, energy)
            img_temp = remove_vertical_seam(img_temp, seam)
            
            # Actualizar progreso
            progress = (i + 1) / (2 * num_seams)
            progress_bar.progress(progress)
            status_text.text(f"Fase 1/2: Seam {i+1}/{num_seams} removido")
            
            # Mostrar proceso si está habilitado
            if mostrar_proceso and i % 5 == 0:
                st.image(bgr_to_rgb(img_temp), caption=f"Seams removidos: {i+1}", 
                        use_container_width=True)
        
        # Fase 2: Rellenar para mantener tamaño original
        status_text.text("Fase 2/2: Rellenando región...")
        
        img_output = img_temp.copy()
        
        for i in range(num_seams):
            energy = compute_energy_matrix(img_temp, energy_method)
            seam = find_vertical_seam(img_temp, energy)
            img_temp = remove_vertical_seam(img_temp, seam)
            img_output = add_vertical_seam(img_output, seam, i)
            
            # Actualizar progreso
            progress = 0.5 + (i + 1) / (2 * num_seams)
            progress_bar.progress(progress)
            status_text.text(f"Fase 2/2: Seam {i+1}/{num_seams} agregado")
            
            # Mostrar proceso si está habilitado
            if mostrar_proceso and i % 5 == 0:
                st.image(bgr_to_rgb(img_output), caption=f"Seams agregados: {i+1}",
                        use_container_width=True)
        
        progress_bar.progress(1.0)
        status_text.text("Procesamiento completado!")
    
    st.success(f"🎉 ¡Objeto eliminado exitosamente! Se procesaron {num_seams * 2} seams en total.")
    
    # Mostrar resultados
    st.markdown("---")
    st.markdown("### Resultado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        mostrar_imagen_streamlit(img, f"Tamaño: {img.shape[1]} x {img.shape[0]}px")
    
    with col2:
        st.markdown("**Objeto Eliminado**")
        mostrar_imagen_streamlit(img_output, f"Tamaño: {img_output.shape[1]} x {img_output.shape[0]}px")
    
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
    
    st.success(f"🎉 Imagen redimensionada de {img.shape[1]}px a {img_result.shape[1]}px")
    
    # Mostrar resultados
    st.markdown("---")
    st.markdown("### Resultado del Redimensionado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Original ({img.shape[1]}px ancho)**")
        mostrar_imagen_streamlit(img, "")
    
    with col2:
        st.markdown(f"**Redimensionado ({img_result.shape[1]}px ancho)**")
        mostrar_imagen_streamlit(img_result, "")
    
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
        
        for i in range(num_seams):
            energy = compute_energy_matrix(img_seam)
            seam = find_vertical_seam(img_seam, energy)
            img_seam = remove_vertical_seam(img_seam, seam)
    
    # Mostrar comparación
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Original ({img.shape[1]}px)**")
        mostrar_imagen_streamlit(img, "")
    
    with col2:
        st.markdown(f"**Resize Tradicional ({nuevo_ancho}px)**")
        mostrar_imagen_streamlit(img_resized, "")
        st.caption("Distorsiona el contenido uniformemente")
    
    with col3:
        st.markdown(f"**Seam Carving ({nuevo_ancho}px)**")
        mostrar_imagen_streamlit(img_seam, "")
        st.caption("Preserva contenido importante")
    
    st.info("""
    **💡 Observa la diferencia:**
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
    
    # Objeto principal (rectángulo azul)
    cv2.rectangle(img, (200, 150), (350, 300), (255, 100, 100), -1)
    cv2.rectangle(img, (200, 150), (350, 300), (200, 50, 50), 3)
    cv2.putText(img, "OBJETO", (230, 235), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 255, 255), 2)
    
    # Objetos de fondo (círculos verdes)
    cv2.circle(img, (100, 100), 40, (100, 255, 100), -1)
    cv2.circle(img, (500, 300), 50, (100, 255, 100), -1)
    
    return img


def crear_imagen_demo_simple():
    """Crea una imagen simple para demostración del proceso técnico."""
    img = np.ones((200, 300, 3), dtype=np.uint8) * 180
    
    # Agregar algunas características
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
            label="⬇Descargar imagen",
            data=file,
            file_name=filename,
            mime="image/jpeg"
        )


if __name__ == "__main__":
    run()