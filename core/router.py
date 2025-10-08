"""
Router/Controlador principal para cargar dinámicamente los módulos de ejercicios
"""
import importlib
from pathlib import Path
from typing import Optional, Dict, List
import streamlit as st


class RouterEjercicio:
    """
    Controlador que gestiona la carga dinámica de ejercicios organizados por capítulos.
    """
    
    def __init__(self):
        """Inicializa el router y descubre los ejercicios disponibles."""
        self.ejercicios_path = Path(__file__).parent.parent / "exercises"
        self.estructura = self._descubrir_ejercicios()
    
    def _descubrir_ejercicios(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Escanea la carpeta exercises/ y construye un diccionario con la estructura:
        {
            "chapter1": [
                {"nombre": "ex01_basic_read", "titulo": "Lectura Básica", "modulo": "exercises.chapter1.ex01_basic_read"},
                ...
            ],
            ...
        }
        
        Returns:
            Diccionario con la estructura de capítulos y ejercicios
        """
        estructura = {}
        
        if not self.ejercicios_path.exists():
            st.warning(f"No se encontró la carpeta de ejercicios: {self.ejercicios_path}")
            return estructura
        
        # Iterar sobre las carpetas de capítulos
        for capitulo_dir in sorted(self.ejercicios_path.iterdir()):
            if not capitulo_dir.is_dir() or capitulo_dir.name.startswith("_"):
                continue
            
            capitulo_nombre = capitulo_dir.name
            ejercicios_lista = []
            
            # Buscar archivos de ejercicios en el capítulo (ex*.py o ejer_*.py)
            ejercicio_files = list(capitulo_dir.glob("ex*.py")) + list(capitulo_dir.glob("ejer_*.py"))
            for ejercicio_file in sorted(ejercicio_files):
                nombre_ejercicio = ejercicio_file.stem
                modulo_path = f"exercises.{capitulo_nombre}.{nombre_ejercicio}"
                
                # Generar un título legible del nombre del archivo
                titulo = self._generar_titulo(nombre_ejercicio)
                
                ejercicios_lista.append({
                    "nombre": nombre_ejercicio,
                    "titulo": titulo,
                    "modulo": modulo_path
                })
            
            if ejercicios_lista:
                estructura[capitulo_nombre] = ejercicios_lista
        
        return estructura
    
    def _generar_titulo(self, nombre_archivo: str) -> str:
        """
        Convierte un nombre de archivo en un título legible.
        Ejemplos:
        - 'ex01_basic_read' -> 'Ex01: Basic Read'
        - 'ejer_transform_proyective' -> 'Transformación Proyectiva'
        
        Args:
            nombre_archivo: Nombre del archivo sin extensión
        
        Returns:
            Título formateado
        """
        # Si empieza con 'ejer_', formato especial
        if nombre_archivo.startswith("ejer_"):
            nombre_archivo = nombre_archivo[5:]  # Remover 'ejer_'
            partes = nombre_archivo.split("_")
            return " ".join(parte.capitalize() for parte in partes)
        
        # Si empieza con 'ex', formato con número
        if nombre_archivo.startswith("ex"):
            nombre_archivo = nombre_archivo[2:]
            partes = nombre_archivo.split("_")
            
            # El primer elemento suele ser el número
            if partes[0].isdigit():
                numero = partes[0]
                resto = " ".join(parte.capitalize() for parte in partes[1:])
                return f"Ex{numero}: {resto}"
        
        # Formato por defecto
        partes = nombre_archivo.split("_")
        return " ".join(parte.capitalize() for parte in partes)
    
    def obtener_capitulos(self) -> List[str]:
        """
        Obtiene la lista de capítulos disponibles.
        
        Returns:
            Lista de nombres de capítulos
        """
        return list(self.estructura.keys())
    
    def obtener_ejercicios(self, capitulo: str) -> List[Dict[str, str]]:
        """
        Obtiene la lista de ejercicios para un capítulo específico.
        
        Args:
            capitulo: Nombre del capítulo
        
        Returns:
            Lista de diccionarios con información de cada ejercicio
        """
        return self.estructura.get(capitulo, [])
    
    def cargar_ejercicio(self, capitulo: str, nombre_ejercicio: str):
        """
        Carga dinámicamente un módulo de ejercicio.
        
        Args:
            capitulo: Nombre del capítulo
            nombre_ejercicio: Nombre del ejercicio (sin extensión .py)
        
        Returns:
            Módulo cargado o None si falla
        """
        ejercicios = self.obtener_ejercicios(capitulo)
        
        # Buscar el ejercicio en la lista
        ejercicio_info = None
        for ej in ejercicios:
            if ej["nombre"] == nombre_ejercicio:
                ejercicio_info = ej
                break
        
        if not ejercicio_info:
            st.error(f"Ejercicio '{nombre_ejercicio}' no encontrado en '{capitulo}'")
            return None
        
        try:
            # Importar el módulo dinámicamente
            modulo = importlib.import_module(ejercicio_info["modulo"])
            
            # Opcional: recargar el módulo para desarrollo (útil en modo debug)
            if st.session_state.get("debug_mode", False):
                importlib.reload(modulo)
            
            return modulo
            
        except ModuleNotFoundError as e:
            st.error(f"No se pudo importar el módulo: {ejercicio_info['modulo']}")
            st.code(str(e))
            return None
        except Exception as e:
            st.error(f"Error al cargar el ejercicio: {str(e)}")
            with st.expander("🔍 Detalles técnicos"):
                st.code(str(e))
            return None
    
    def obtener_titulo_ejercicio(self, capitulo: str, nombre_ejercicio: str) -> Optional[str]:
        """
        Obtiene el título formateado de un ejercicio específico.
        
        Args:
            capitulo: Nombre del capítulo
            nombre_ejercicio: Nombre del ejercicio
        
        Returns:
            Título del ejercicio o None si no se encuentra
        """
        ejercicios = self.obtener_ejercicios(capitulo)
        
        for ej in ejercicios:
            if ej["nombre"] == nombre_ejercicio:
                return ej["titulo"]
        
        return None
    
    def tiene_ejercicios(self) -> bool:
        """
        Verifica si hay ejercicios disponibles.
        
        Returns:
            True si hay al menos un ejercicio, False en caso contrario
        """
        return len(self.estructura) > 0
    
    def obtener_estadisticas(self) -> Dict[str, int]:
        """
        Obtiene estadísticas sobre los ejercicios disponibles.
        
        Returns:
            Diccionario con estadísticas
        """
        total_ejercicios = sum(len(ejercicios) for ejercicios in self.estructura.values())
        
        return {
            "total_capitulos": len(self.estructura),
            "total_ejercicios": total_ejercicios,
            "ejercicios_por_capitulo": {
                capitulo: len(ejercicios)
                for capitulo, ejercicios in self.estructura.items()
            }
        }