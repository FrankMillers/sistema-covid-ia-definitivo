"""
Sistema de Multilenguaje para Aplicación COVID-19 IA
Autor: Sistema IA COVID-19
Descripción: Gestiona traducciones en Español, Inglés y Francés
"""

import json
import os
import streamlit as st
from typing import Dict, Any

class GestorMultilenguaje:
    """Gestiona las traducciones y cambios de idioma en la aplicación"""
    
    def __init__(self):
        """Inicializa el gestor de multilenguaje"""
        self.idiomas_disponibles = {
            'es': '🇪🇸 Español',
            'en': '🇬🇧 English', 
            'fr': '🇫🇷 Français'
        }
        self.idioma_por_defecto = 'es'
        self.traducciones = self._cargar_todas_las_traducciones()
        
    def _cargar_todas_las_traducciones(self) -> Dict[str, Dict[str, Any]]:
        """Carga todos los archivos de traducción disponibles"""
        traducciones = {}
        ruta_traducciones = 'traducciones'
        
        # Crear directorio si no existe
        if not os.path.exists(ruta_traducciones):
            os.makedirs(ruta_traducciones)
            
        # Cargar cada idioma disponible
        for codigo_idioma in self.idiomas_disponibles.keys():
            archivo_traduccion = os.path.join(ruta_traducciones, f'{codigo_idioma}.json')
            
            try:
                if os.path.exists(archivo_traduccion):
                    with open(archivo_traduccion, 'r', encoding='utf-8') as archivo:
                        traducciones[codigo_idioma] = json.load(archivo)
                        print(f"✅ Traducciones {codigo_idioma.upper()} cargadas correctamente")
                else:
                    print(f"⚠️ Archivo de traducción {archivo_traduccion} no encontrado")
                    traducciones[codigo_idioma] = {}
                    
            except Exception as error:
                print(f"❌ Error cargando traducciones {codigo_idioma}: {str(error)}")
                traducciones[codigo_idioma] = {}
                
        return traducciones
    
    def obtener_idioma_actual(self) -> str:
        """Obtiene el idioma actual de la sesión con manejo robusto"""
        try:
            # Verificar si existe contexto de Streamlit
            if hasattr(st, 'session_state'):
                # Inicializar si no existe
                if 'idioma_seleccionado' not in st.session_state:
                    st.session_state.idioma_seleccionado = self.idioma_por_defecto
                return st.session_state.idioma_seleccionado
            else:
                # No hay contexto de Streamlit, usar por defecto
                return self.idioma_por_defecto
        except Exception:
            # En caso de cualquier error, usar idioma por defecto
            return self.idioma_por_defecto
    
    def cambiar_idioma(self, nuevo_idioma: str) -> None:
        """Cambia el idioma actual de la aplicación"""
        try:
            if nuevo_idioma in self.idiomas_disponibles:
                if hasattr(st, 'session_state'):
                    st.session_state.idioma_seleccionado = nuevo_idioma
                    # Usar st.rerun() en lugar de deprecated st.experimental_rerun()
                    try:
                        st.rerun()
                    except AttributeError:
                        # Fallback para versiones anteriores de Streamlit
                        st.experimental_rerun()
        except Exception as e:
            print(f"⚠️ Error cambiando idioma: {e}")
    
    def traducir(self, clave: str, idioma: str = None) -> str:
        """
        Traduce una clave al idioma especificado o actual
        
        Args:
            clave (str): Clave de traducción a buscar
            idioma (str, optional): Código de idioma. Si no se especifica, usa el actual
            
        Returns:
            str: Texto traducido o la clave original si no se encuentra
        """
        try:
            if idioma is None:
                idioma = self.obtener_idioma_actual()
                
            # Verificar que el idioma existe
            if idioma not in self.traducciones:
                idioma = self.idioma_por_defecto
                
            # Verificar que tenemos traducciones para el idioma
            if not self.traducciones.get(idioma):
                return clave
                
            # Manejar claves anidadas (ej: "eda.configuracion")
            if '.' in clave:
                partes = clave.split('.')
                traduccion_actual = self.traducciones[idioma]
                
                try:
                    for parte in partes:
                        if isinstance(traduccion_actual, dict) and parte in traduccion_actual:
                            traduccion_actual = traduccion_actual[parte]
                        else:
                            # Si no se encuentra la clave anidada, intentar con idioma por defecto
                            if idioma != self.idioma_por_defecto and self.traducciones.get(self.idioma_por_defecto):
                                traduccion_defecto = self.traducciones[self.idioma_por_defecto]
                                for parte_def in partes:
                                    if isinstance(traduccion_defecto, dict) and parte_def in traduccion_defecto:
                                        traduccion_defecto = traduccion_defecto[parte_def]
                                    else:
                                        return clave  # Devolver clave original si no se encuentra
                                if isinstance(traduccion_defecto, str):
                                    return traduccion_defecto
                            return clave  # Devolver clave original si no se encuentra
                    
                    # Si llegamos aquí, encontramos la traducción
                    if isinstance(traduccion_actual, str):
                        return traduccion_actual
                    else:
                        return clave
                        
                except (KeyError, TypeError):
                    return clave
                    
            else:
                # Clave simple (sin puntos)
                traduccion = self.traducciones[idioma].get(clave, clave)
                
                # Si no se encuentra en el idioma actual, intentar con el por defecto
                if traduccion == clave and idioma != self.idioma_por_defecto and self.traducciones.get(self.idioma_por_defecto):
                    traduccion = self.traducciones[self.idioma_por_defecto].get(clave, clave)
                    
                return traduccion
                
        except Exception as e:
            print(f"⚠️ Error traduciendo clave '{clave}': {e}")
            return clave
    
    def t(self, clave: str) -> str:
        """Método abreviado para traducir"""
        return self.traducir(clave)
    
    def crear_selector_idioma_sidebar(self) -> str:
        """
        Crea un selector de idioma en el sidebar de Streamlit
        
        Returns:
            str: Código del idioma seleccionado
        """
        try:
            # Verificar si estamos en contexto de Streamlit
            if not hasattr(st, 'sidebar'):
                return self.idioma_por_defecto
                
            st.sidebar.markdown("---")
            
            # Asegurar que el idioma actual esté inicializado
            idioma_actual = self.obtener_idioma_actual()
            
            # Verificar que el idioma actual está en la lista de opciones
            if idioma_actual not in self.idiomas_disponibles:
                idioma_actual = self.idioma_por_defecto
                st.session_state.idioma_seleccionado = idioma_actual
            
            # Selector de idioma
            idioma_seleccionado = st.sidebar.selectbox(
                label=self.t("idioma"),
                options=list(self.idiomas_disponibles.keys()),
                index=list(self.idiomas_disponibles.keys()).index(idioma_actual),
                format_func=lambda x: self.idiomas_disponibles[x],
                help=self.t("seleccionar_idioma"),
                key="selector_idioma_multilenguaje"
            )
            
            # Detectar cambio de idioma
            if idioma_seleccionado != idioma_actual:
                self.cambiar_idioma(idioma_seleccionado)
                
            return idioma_seleccionado
            
        except Exception as e:
            print(f"⚠️ Error creando selector de idioma: {e}")
            return self.idioma_por_defecto
    
    def obtener_traducciones_modelo(self) -> Dict[str, str]:
        """Obtiene las traducciones específicas para los modelos"""
        try:
            idioma_actual = self.obtener_idioma_actual()
            
            # Verificar que el idioma existe y tiene traducciones
            if idioma_actual not in self.traducciones or not self.traducciones[idioma_actual]:
                idioma_actual = self.idioma_por_defecto
                
            modelos_dict = self.traducciones[idioma_actual].get('modelos_disponibles', {})
            
            # Si no existen traducciones para modelos, usar valores por defecto
            if not modelos_dict:
                modelos_dict = {
                    "Custom_CNN": "🏆 Campeón",
                    "MobileNetV2": "🥈 Segundo", 
                    "Ensemble": "🎯 Ensemble",
                    "EfficientNet": "📊 EfficientNet",
                    "CNN_XGBoost": "🔗 Híbrido XGB",
                    "CNN_RandomForest": "🌳 Híbrido RF"
                }
                
            return modelos_dict
            
        except Exception as e:
            print(f"⚠️ Error obteniendo traducciones de modelos: {e}")
            return {
                "Custom_CNN": "🏆 Campeón",
                "MobileNetV2": "🥈 Segundo", 
                "Ensemble": "🎯 Ensemble",
                "EfficientNet": "📊 EfficientNet",
                "CNN_XGBoost": "🔗 Híbrido XGB",
                "CNN_RandomForest": "🌳 Híbrido RF"
            }
    
    def formatear_texto_con_valores(self, clave: str, **valores) -> str:
        """
        Formatea un texto traducido con valores dinámicos
        
        Args:
            clave (str): Clave de traducción
            **valores: Valores para formatear en el texto
            
        Returns:
            str: Texto formateado
        """
        texto_base = self.traducir(clave)
        
        try:
            return texto_base.format(**valores)
        except (KeyError, ValueError) as error:
            print(f"⚠️ Error formateando texto '{clave}': {error}")
            return texto_base
    
    def obtener_estadisticas_traducciones(self) -> Dict[str, Any]:
        """Obtiene estadísticas sobre las traducciones cargadas"""
        estadisticas = {
            'idiomas_cargados': len([k for k, v in self.traducciones.items() if v]),
            'idioma_actual': self.obtener_idioma_actual(),
            'claves_por_idioma': {}
        }
        
        for idioma, traducciones in self.traducciones.items():
            if isinstance(traducciones, dict):
                estadisticas['claves_por_idioma'][idioma] = len(traducciones)
            else:
                estadisticas['claves_por_idioma'][idioma] = 0
            
        return estadisticas
    
    def verificar_integridad_traducciones(self) -> Dict[str, Any]:
        """Verifica que todas las traducciones tengan las mismas claves"""
        if not self.traducciones or not self.traducciones.get(self.idioma_por_defecto):
            return {'valido': False, 'errores': ['No hay traducciones cargadas']}
            
        # Obtener claves del idioma por defecto como referencia
        claves_referencia = set(self.traducciones[self.idioma_por_defecto].keys())
        errores = []
        
        for idioma, traducciones in self.traducciones.items():
            if idioma == self.idioma_por_defecto or not traducciones:
                continue
                
            claves_idioma = set(traducciones.keys())
            
            # Claves faltantes
            claves_faltantes = claves_referencia - claves_idioma
            if claves_faltantes:
                errores.append(f"Idioma {idioma}: faltan claves {list(claves_faltantes)}")
                
            # Claves extra
            claves_extra = claves_idioma - claves_referencia
            if claves_extra:
                errores.append(f"Idioma {idioma}: claves extra {list(claves_extra)}")
        
        return {
            'valido': len(errores) == 0,
            'errores': errores,
            'total_claves_referencia': len(claves_referencia)
        }

# Crear instancia global del gestor
gestor_multilenguaje = GestorMultilenguaje()

# Funciones de conveniencia para usar en la aplicación
def t(clave: str) -> str:
    """Función global para traducir texto"""
    return gestor_multilenguaje.traducir(clave)

def cambiar_idioma(nuevo_idioma: str) -> None:
    """Función global para cambiar idioma"""
    gestor_multilenguaje.cambiar_idioma(nuevo_idioma)

def crear_selector_idioma() -> str:
    """Función global para crear selector de idioma"""
    return gestor_multilenguaje.crear_selector_idioma_sidebar()

def obtener_idioma_actual() -> str:
    """Función global para obtener idioma actual"""
    return gestor_multilenguaje.obtener_idioma_actual()

def obtener_traducciones_modelos() -> Dict[str, str]:
    """Función global para obtener traducciones de modelos"""
    return gestor_multilenguaje.obtener_traducciones_modelo()