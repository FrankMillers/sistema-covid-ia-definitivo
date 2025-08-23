"""
Módulo de Generación de Reportes PDF Multilenguaje
Sistema COVID-19 IA

Este módulo proporciona funcionalidades completas para generar reportes PDF 
en múltiples idiomas (ES/EN/FR) para todos los dashboards del sistema.

Características principales:
- Generación de reportes profesionales en PDF
- Soporte multilenguaje completo
- Reportes específicos para cada dashboard
- Integración con sistema de traducciones
- Formateo automático según idioma

Módulos disponibles:
- generador_reportes: Clase principal para generación de PDFs
"""

__version__ = "1.0.0"
__author__ = "Sistema COVID-19 IA"
__description__ = "Módulo de generación de reportes PDF multilenguaje"

# Importaciones principales
try:
    from .generador_reportes import GeneradorReportesPDF, generador_reportes
    
    __all__ = [
        'GeneradorReportesPDF',
        'generador_reportes'
    ]
    
    # Verificar dependencias
    try:
        import reportlab
        REPORTLAB_DISPONIBLE = True
    except ImportError:
        REPORTLAB_DISPONIBLE = False
        print("⚠️ ReportLab no disponible. Instale con: pip install reportlab")
    
    try:
        import plotly
        PLOTLY_DISPONIBLE = True
    except ImportError:
        PLOTLY_DISPONIBLE = False
        print("⚠️ Plotly no disponible. Instale con: pip install plotly")
    
    # Status del módulo
    if REPORTLAB_DISPONIBLE and PLOTLY_DISPONIBLE:
        MODULO_COMPLETO = True
        print("✅ Módulo de reportes PDF completamente funcional")
    else:
        MODULO_COMPLETO = False
        print("⚠️ Módulo de reportes PDF con funcionalidad limitada")

except ImportError as e:
    print(f"❌ Error importando módulo de reportes PDF: {e}")
    print("💡 Instale las dependencias: pip install reportlab plotly pandas numpy")
    
    # Crear versiones dummy para evitar errores
    class GeneradorReportesPDF:
        def __init__(self):
            pass
        
        def generar_reporte_analisis_individual(self, *args, **kwargs):
            raise ImportError("Módulo de reportes PDF no disponible")
        
        def generar_reporte_eda(self, *args, **kwargs):
            raise ImportError("Módulo de reportes PDF no disponible")
        
        def generar_reporte_comparacion_modelos(self, *args, **kwargs):
            raise ImportError("Módulo de reportes PDF no disponible")
    
    generador_reportes = GeneradorReportesPDF()
    
    __all__ = ['GeneradorReportesPDF', 'generador_reportes']
    MODULO_COMPLETO = False

# Información de utilidad
def obtener_info_modulo():
    """Retorna información sobre el estado del módulo"""
    return {
        'version': __version__,
        'autor': __author__,
        'descripcion': __description__,
        'modulo_completo': MODULO_COMPLETO,
        'dependencias_instaladas': {
            'reportlab': REPORTLAB_DISPONIBLE if 'REPORTLAB_DISPONIBLE' in locals() else False,
            'plotly': PLOTLY_DISPONIBLE if 'PLOTLY_DISPONIBLE' in locals() else False
        }
    }

def verificar_dependencias():
    """Verifica e instala dependencias faltantes"""
    dependencias_faltantes = []
    
    try:
        import reportlab
    except ImportError:
        dependencias_faltantes.append("reportlab")
    
    try:
        import plotly
    except ImportError:
        dependencias_faltantes.append("plotly")
    
    if dependencias_faltantes:
        print("❌ Dependencias faltantes para reportes PDF:")
        for dep in dependencias_faltantes:
            print(f"   - {dep}")
        print("\n💡 Instale con:")
        print(f"   pip install {' '.join(dependencias_faltantes)}")
        return False
    else:
        print("✅ Todas las dependencias están instaladas")
        return True

# Configuración de tipos de reporte disponibles
TIPOS_REPORTE = {
    'analisis': {
        'nombre': 'Análisis Individual de Radiografía',
        'descripcion': 'Reporte detallado del análisis de una radiografía específica',
        'idiomas': ['es', 'en', 'fr']
    },
    'eda': {
        'nombre': 'Análisis Exploratorio de Datos',
        'descripcion': 'Reporte estadístico completo del dataset COVID-19',
        'idiomas': ['es', 'en', 'fr']
    },
    'comparacion': {
        'nombre': 'Comparación Robusta de Modelos',
        'descripcion': 'Evaluación comparativa de modelos de IA COVID-19',
        'idiomas': ['es', 'en', 'fr']
    }
}

def obtener_tipos_reporte_disponibles():
    """Retorna los tipos de reporte disponibles"""
    return TIPOS_REPORTE

# Configuraciones por defecto
CONFIG_DEFAULT = {
    'formato_fecha': {
        'es': '%d de %B de %Y',
        'en': '%B %d, %Y',
        'fr': '%d %B %Y'
    },
    'tamaño_pagina': 'A4',
    'margenes': {
        'superior': 72,
        'inferior': 18,
        'izquierdo': 72,
        'derecho': 72
    },
    'fuentes': {
        'titulo': 'Helvetica-Bold',
        'subtitulo': 'Helvetica-Bold',
        'texto': 'Helvetica',
        'destacado': 'Helvetica-Bold'
    }
}

def obtener_configuracion_default():
    """Retorna configuración por defecto para reportes"""
    return CONFIG_DEFAULT