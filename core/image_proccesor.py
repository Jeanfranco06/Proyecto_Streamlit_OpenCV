"""
Clase centralizada para procesamiento de imágenes
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .validators import validate_image, validate_kernel_size


class ImageProcessor:
    """Clase para operaciones comunes de procesamiento de imágenes"""
    
    def __init__(self, image: np.ndarray):
        """
        Inicializa el procesador con una imagen
        
        Args:
            image: Imagen en formato numpy array
        """
        self.original = image.copy()
        self.current = image.copy()
        self.history = [image.copy()]
    
    def reset(self):
        """Resetea la imagen a su estado original"""
        self.current = self.original.copy()
        self.history = [self.original.copy()]
    
    def undo(self):
        """Deshace la última operación"""
        if len(self.history) > 1:
            self.history.pop()
            self.current = self.history[-1].copy()
    
    def get_current(self) -> np.ndarray:
        """Retorna la imagen actual"""
        return self.current.copy()
    
    def get_original(self) -> np.ndarray:
        """Retorna la imagen original"""
        return self.original.copy()
    
    def to_grayscale(self) -> np.ndarray:
        """Convierte la imagen actual a escala de grises"""
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
            self.current = gray
            self.history.append(gray.copy())
        return self.current
    
    def resize(self, width: int, height: int, 
               keep_aspect: bool = True) -> np.ndarray:
        """
        Redimensiona la imagen
        
        Args:
            width: Ancho objetivo
            height: Alto objetivo
            keep_aspect: Mantener relación de aspecto
        
        Returns:
            Imagen redimensionada
        """
        h, w = self.current.shape[:2]
        
        if keep_aspect:
            ratio = min(width / w, height / h)
            new_size = (int(w * ratio), int(h * ratio))
        else:
            new_size = (width, height)
        
        resized = cv2.resize(self.current, new_size, interpolation=cv2.INTER_AREA)
        self.current = resized
        self.history.append(resized.copy())
        return self.current
    
    def blur(self, kernel_size: int = 5, method: str = 'gaussian') -> np.ndarray:
        """
        Aplica desenfoque a la imagen
        
        Args:
            kernel_size: Tamaño del kernel (debe ser impar)
            method: Método ('gaussian', 'median', 'bilateral')
        
        Returns:
            Imagen desenfocada
        """
        kernel_size = validate_kernel_size(kernel_size)
        
        if method == 'gaussian':
            blurred = cv2.GaussianBlur(self.current, (kernel_size, kernel_size), 0)
        elif method == 'median':
            blurred = cv2.medianBlur(self.current, kernel_size)
        elif method == 'bilateral':
            blurred = cv2.bilateralFilter(self.current, kernel_size, 75, 75)
        else:
            blurred = cv2.blur(self.current, (kernel_size, kernel_size))
        
        self.current = blurred
        self.history.append(blurred.copy())
        return self.current
    
    def edge_detection(self, method: str = 'canny', 
                       threshold1: int = 100, 
                       threshold2: int = 200) -> np.ndarray:
        """
        Detecta bordes en la imagen
        
        Args:
            method: Método ('canny', 'sobel', 'laplacian')
            threshold1: Primer umbral
            threshold2: Segundo umbral
        
        Returns:
            Imagen con bordes detectados
        """
        # Convertir a escala de grises si es necesario
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current
        
        if method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.uint8(np.sqrt(sobelx**2 + sobely**2))
        elif method == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            edges = gray
        
        self.current = edges
        self.history.append(edges.copy())
        return self.current
    
    def threshold(self, thresh_value: int = 127, 
                  method: str = 'binary') -> np.ndarray:
        """
        Aplica umbralización a la imagen
        
        Args:
            thresh_value: Valor de umbral
            method: Método ('binary', 'otsu', 'adaptive')
        
        Returns:
            Imagen umbralizada
        """
        # Convertir a escala de grises si es necesario
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current
        
        if method == 'binary':
            _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        elif method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            thresh = gray
        
        self.current = thresh
        self.history.append(thresh.copy())
        return self.current
    
    def morphology(self, operation: str = 'erode', 
                   kernel_size: int = 5) -> np.ndarray:
        """
        Aplica operaciones morfológicas
        
        Args:
            operation: Operación ('erode', 'dilate', 'open', 'close')
            kernel_size: Tamaño del kernel
        
        Returns:
            Imagen procesada
        """
        kernel_size = validate_kernel_size(kernel_size)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erode':
            result = cv2.erode(self.current, kernel, iterations=1)
        elif operation == 'dilate':
            result = cv2.dilate(self.current, kernel, iterations=1)
        elif operation == 'open':
            result = cv2.morphologyEx(self.current, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            result = cv2.morphologyEx(self.current, cv2.MORPH_CLOSE, kernel)
        else:
            result = self.current
        
        self.current = result
        self.history.append(result.copy())
        return self.current
    
    def find_contours(self) -> Tuple[List, np.ndarray]:
        """
        Encuentra contornos en la imagen
        
        Returns:
            Tupla (contornos, jerarquía)
        """
        # Convertir a escala de grises si es necesario
        if len(self.current.shape) == 3:
            gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current
        
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours, hierarchy
    
    def draw_contours(self, contours: List, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """
        Dibuja contornos sobre la imagen
        
        Args:
            contours: Lista de contornos
            color: Color BGR
            thickness: Grosor de línea
        
        Returns:
            Imagen con contornos dibujados
        """
        # Asegurar que sea imagen de color
        if len(self.current.shape) == 2:
            result = cv2.cvtColor(self.current, cv2.COLOR_GRAY2BGR)
        else:
            result = self.current.copy()
        
        cv2.drawContours(result, contours, -1, color, thickness)
        
        self.current = result
        self.history.append(result.copy())
        return self.current
    
    def get_histogram(self, channel: Optional[int] = None) -> np.ndarray:
        """
        Calcula el histograma de la imagen
        
        Args:
            channel: Canal específico (0=B, 1=G, 2=R) o None para escala de grises
        
        Returns:
            Histograma
        """
        if len(self.current.shape) == 2:
            # Imagen en escala de grises
            hist = cv2.calcHist([self.current], [0], None, [256], [0, 256])
        else:
            if channel is not None and 0 <= channel < 3:
                hist = cv2.calcHist([self.current], [channel], None, [256], [0, 256])
            else:
                # Convertir a escala de grises
                gray = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        return hist
    
    def equalize_histogram(self) -> np.ndarray:
        """
        Ecualiza el histograma de la imagen
        
        Returns:
            Imagen ecualizada
        """
        if len(self.current.shape) == 2:
            equalized = cv2.equalizeHist(self.current)
        else:
            # Convertir a YCrCb y ecualizar el canal Y
            ycrcb = cv2.cvtColor(self.current, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        self.current = equalized
        self.history.append(equalized.copy())
        return self.current