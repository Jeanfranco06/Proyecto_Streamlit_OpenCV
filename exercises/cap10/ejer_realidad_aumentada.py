import streamlit as st

import cv2
import numpy as np
from PIL import Image

st.title("Realidad Aumentada - AR Detector")

st.warning("""
⚠️ **Nota importante:**
- En **desarrollo local**: Funciona con cámara web en vivo
- En **Streamlit Cloud**: Sube imagen/video para procesar

Para usar cámara en vivo en producción, ejecuta localmente:
```bash
streamlit run ar_ejercicio10.py
```
""")

# Opción 1: Cámara local
col1, col2 = st.columns(2)
with col1:
    usar_camara = st.checkbox("Usar cámara (solo local)", value=False)

with col2:
    usar_archivo = st.checkbox("Subir imagen/video", value=True)

if usar_camara:
    st.subheader("📹 Cámara en vivo")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("❌ No se pudo acceder a la cámara. Verifica que esté conectada.")
    else:
        frame_placeholder = st.empty()
        stop_button = st.button("Detener cámara")
        
        while not stop_button:
            ret, frame = camera.read()
            if not ret:
                st.error("Error al capturar frame")
                break
            
            # Redimensionar para mostrar
            frame = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Mostrar
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            if st.session_state.get("stop_camera"):
                break
        
        camera.release()
        st.info("Cámara cerrada")

elif usar_archivo:
    st.subheader("📁 Subir archivo")
    
    archivo = st.file_uploader(
        "Selecciona imagen o video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )
    
    if archivo is not None:
        # Detecta tipo
        if archivo.type.startswith("image"):
            img = Image.open(archivo)
            st.image(img, caption="Imagen cargada", use_column_width=True)
            st.success("✓ Imagen cargada")
        
        elif archivo.type.startswith("video"):
            st.info("📹 Video cargado (vista previa en construcción)")
            st.success("✓ Video cargado")
        
        else:
            st.error("Tipo de archivo no soportado")

else:
    st.info("Selecciona una opción para comenzar")