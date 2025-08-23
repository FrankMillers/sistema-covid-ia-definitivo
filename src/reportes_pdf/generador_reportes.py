"""
Sistema de Generación de Reportes PDF Multilenguaje - MEJORADO
Autor: Sistema IA COVID-19
Descripción: Genera reportes PDF profesionales con espaciado mejorado
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib import colors
import io
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os
import tempfile

# Importar sistema multilenguaje
sys.path.append('src')
try:
    from sistema_multilenguaje.sistema_multilenguaje import t, gestor_multilenguaje
    MULTILENGUAJE_DISPONIBLE = True
except ImportError:
    MULTILENGUAJE_DISPONIBLE = False
    def t(clave): return clave

class GeneradorReportesPDF:
    """Generador de reportes PDF multilenguaje con espaciado profesional"""
    
    def __init__(self):
        """Inicializa el generador con estilos médicos profesionales"""
        # Configurar colores médicos profesionales
        self.colores = {
            # Colores principales médicos
            'primario': HexColor('#1E88E5'),        # Azul médico profesional
            'secundario': HexColor('#43A047'),       # Verde médico
            'acento': HexColor('#FF8F00'),          # Naranja profesional
            'texto': HexColor('#212121'),           # Negro suave
            'fondo': HexColor('#FAFAFA'),           # Gris muy claro
            
            # Colores de diagnóstico
            'covid_critico': HexColor('#D32F2F'),    # Rojo intenso para COVID
            'covid_alerta': HexColor('#F57C00'),     # Naranja para sospechoso
            'normal': HexColor('#388E3C'),           # Verde para normal
            'opacidad': HexColor('#1976D2'),         # Azul para opacidades
            'neumonia': HexColor('#7B1FA2'),         # Púrpura para neumonía
            
            # Colores de estado
            'excelente': HexColor('#4CAF50'),        # Verde brillante
            'bueno': HexColor('#8BC34A'),            # Verde claro
            'aceptable': HexColor('#FFC107'),        # Amarillo
            'pobre': HexColor('#FF9800'),            # Naranja
            'critico': HexColor('#F44336'),          # Rojo
            
            # Colores de fondo para destacar
            'fondo_covid': HexColor('#FFEBEE'),      # Fondo rojo muy claro
            'fondo_normal': HexColor('#E8F5E8'),     # Fondo verde muy claro
            'fondo_alerta': HexColor('#FFF3E0'),     # Fondo naranja muy claro
            'fondo_info': HexColor('#E3F2FD'),       # Fondo azul muy claro
        }
        
        # Configurar estilos con espaciado mejorado
        self.estilos = self._configurar_estilos_mejorados()
        
    def _configurar_estilos_mejorados(self):
        """Configura estilos con espaciado profesional mejorado"""
        estilos = getSampleStyleSheet()
        
        # Estilo título principal médico con más espaciado
        estilos.add(ParagraphStyle(
            name='TituloPrincipal',
            parent=estilos['Title'],
            fontSize=26,
            spaceAfter=30,      # Aumentado de 25 a 30
            spaceBefore=10,     # Añadido espaciado antes
            textColor=self.colores['primario'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=self.colores['primario'],
            borderPadding=15,   # Aumentado de 12 a 15
            leftIndent=10,
            rightIndent=10
        ))
        
        # Estilo subtítulo médico con mejor espaciado
        estilos.add(ParagraphStyle(
            name='Subtitulo',
            parent=estilos['Heading1'],
            fontSize=18,
            spaceAfter=25,      # Aumentado de 18 a 25
            spaceBefore=10,     # Añadido espaciado antes
            textColor=self.colores['secundario'],
            fontName='Helvetica-Bold',
            leftIndent=15,      # Aumentado de 10 a 15
            borderWidth=1,
            borderColor=self.colores['secundario'],
            borderPadding=10,   # Aumentado de 6 a 10
        ))
        
        # Estilo sección con fondo y mejor espaciado
        estilos.add(ParagraphStyle(
            name='Seccion',
            parent=estilos['Heading2'],
            fontSize=16,
            spaceAfter=18,      # Aumentado de 12 a 18
            spaceBefore=15,     # Añadido espaciado antes
            textColor=self.colores['primario'],
            fontName='Helvetica-Bold',
            backColor=self.colores['fondo_info'],
            borderWidth=1,
            borderColor=self.colores['primario'],
            borderPadding=12,   # Aumentado de 8 a 12
            leftIndent=8,       # Aumentado de 5 a 8
            rightIndent=8       # Añadido margen derecho
        ))
        
        # Estilo texto médico profesional con mejor espaciado
        estilos.add(ParagraphStyle(
            name='TextoPrincipal',
            parent=estilos['Normal'],
            fontSize=12,
            spaceAfter=15,      # Aumentado de 10 a 15
            spaceBefore=5,      # Añadido espaciado antes
            textColor=self.colores['texto'],
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leading=18,         # Aumentado de 16 a 18
            leftIndent=10,      # Añadido margen izquierdo
            rightIndent=10      # Añadido margen derecho
        ))
        
        # Estilo alerta médica crítica con espaciado mejorado
        estilos.add(ParagraphStyle(
            name='AlertaCritica',
            parent=estilos['Normal'],
            fontSize=14,
            spaceAfter=20,      # Aumentado de 15 a 20
            spaceBefore=15,     # Añadido espaciado antes
            textColor=self.colores['covid_critico'],
            fontName='Helvetica-Bold',
            backColor=self.colores['fondo_covid'],
            borderColor=self.colores['covid_critico'],
            borderWidth=2,
            borderPadding=18,   # Aumentado de 12 a 18
            alignment=TA_CENTER,
            leftIndent=15,      # Añadido margen izquierdo
            rightIndent=15      # Añadido margen derecho
        ))
        
        # Estilo resultado normal con mejor espaciado
        estilos.add(ParagraphStyle(
            name='ResultadoNormal',
            parent=estilos['Normal'],
            fontSize=13,
            spaceAfter=18,      # Aumentado de 12 a 18
            spaceBefore=15,     # Añadido espaciado antes
            textColor=self.colores['normal'],
            fontName='Helvetica-Bold',
            backColor=self.colores['fondo_normal'],
            borderColor=self.colores['normal'],
            borderWidth=2,
            borderPadding=15,   # Aumentado de 10 a 15
            alignment=TA_CENTER,
            leftIndent=15,      # Añadido margen izquierdo
            rightIndent=15      # Añadido margen derecho
        ))
        
        # Estilo advertencia con espaciado mejorado
        estilos.add(ParagraphStyle(
            name='Advertencia',
            parent=estilos['Normal'],
            fontSize=12,
            spaceAfter=15,      # Aumentado de 10 a 15
            spaceBefore=10,     # Añadido espaciado antes
            textColor=self.colores['covid_alerta'],
            fontName='Helvetica-Bold',
            backColor=self.colores['fondo_alerta'],
            borderColor=self.colores['covid_alerta'],
            borderWidth=1,
            borderPadding=12,   # Aumentado de 8 a 12
            leftIndent=10,      # Añadido margen izquierdo
            rightIndent=10      # Añadido margen derecho
        ))
        
        # Estilo información médica con espaciado mejorado
        estilos.add(ParagraphStyle(
            name='InfoMedica',
            parent=estilos['Normal'],
            fontSize=11,
            spaceAfter=12,      # Aumentado de 8 a 12
            spaceBefore=8,      # Añadido espaciado antes
            textColor=self.colores['texto'],
            fontName='Helvetica',
            backColor=self.colores['fondo_info'],
            borderColor=self.colores['primario'],
            borderWidth=1,
            borderPadding=12,   # Aumentado de 8 a 12
            leftIndent=15,      # Aumentado de 10 a 15
            rightIndent=15      # Añadido margen derecho
        ))
        
        return estilos
    
    def _obtener_fecha_localizada(self) -> str:
        """Obtiene fecha actual formateada según el idioma"""
        ahora = datetime.now()
        idioma = gestor_multilenguaje.obtener_idioma_actual() if MULTILENGUAJE_DISPONIBLE else 'es'
        
        meses = {
            'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                   'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'],
            'en': ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December'],
            'fr': ['janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                   'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
        }
        
        mes_nombres = meses.get(idioma, meses['es'])
        
        if idioma == 'en':
            return f"{mes_nombres[ahora.month-1]} {ahora.day}, {ahora.year}"
        elif idioma == 'fr':
            return f"{ahora.day} {mes_nombres[ahora.month-1]} {ahora.year}"
        else:  # español
            return f"{ahora.day} de {mes_nombres[ahora.month-1]} de {ahora.year}"
    
    def _crear_encabezado(self, elementos: List, titulo: str, subtitulo: str = ""):
        """Crea encabezado estándar con espaciado mejorado"""
        # Título principal con espaciado mejorado
        elementos.append(Paragraph(titulo, self.estilos['TituloPrincipal']))
        elementos.append(Spacer(1, 20))  # Espaciado después del título
        
        if subtitulo:
            elementos.append(Paragraph(subtitulo, self.estilos['Subtitulo']))
            elementos.append(Spacer(1, 15))  # Espaciado después del subtítulo
        
        # Información del reporte con espaciado mejorado
        fecha = self._obtener_fecha_localizada()
        idioma_actual = gestor_multilenguaje.obtener_idioma_actual() if MULTILENGUAJE_DISPONIBLE else 'es'
        
        info_reporte = f"""
        <para align="center" spaceBefore="10" spaceAfter="10">
        <font size="11" color="#1E88E5">
        <b>📋 {t('reportes.generado_el')}: {fecha}</b><br/><br/>
        <font color="#43A047">🌐 {t('reportes.idioma')}: {idioma_actual.upper()}</font><br/><br/>
        <font color="#7B1FA2">🦠 {t('reportes.sistema')}: COVID-19 IA v2.0</font>
        </font>
        </para>
        """
        
        info_style = ParagraphStyle(
            'InfoReporte',
            parent=self.estilos['Normal'],
            backColor=self.colores['fondo'],
            borderColor=self.colores['primario'],
            borderWidth=1,
            borderPadding=18,   # Aumentado padding
            spaceBefore=15,     # Aumentado espaciado antes
            spaceAfter=20,      # Aumentado espaciado después
            leftIndent=10,      # Añadido margen izquierdo
            rightIndent=10      # Añadido margen derecho
        )
        
        elementos.append(Paragraph(info_reporte, info_style))
        elementos.append(Spacer(1, 30))  # Mayor espaciado después de la info
        
        # Línea separadora médica profesional con espaciado
        separador = f"""
        <para align="center" spaceBefore="10" spaceAfter="15">
        <font color="#1E88E5" size="8">
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        </font>
        </para>
        """
        elementos.append(Paragraph(separador, self.estilos['Normal']))
        elementos.append(Spacer(1, 25))  # Aumentado espaciado después del separador
    
    def _crear_pie_pagina(self, elementos: List):
        """Crea pie de página estándar con espaciado mejorado"""
        elementos.append(Spacer(1, 40))  # Mayor espaciado antes del pie
        
        # Disclaimer médico con colores profesionales y mejor espaciado
        disclaimer = f"""
        <para align="center" spaceBefore="15" spaceAfter="15">
        <font size="12" color="#D32F2F">
        <b>🚨 {t('reportes.disclaimer_titulo')} 🚨</b>
        </font>
        <br/><br/>
        <font size="10" color="#1E88E5">
        {t('reportes.disclaimer_contenido')}
        </font>
        </para>
        """
        
        disclaimer_style = ParagraphStyle(
            'DisclaimerMedico',
            parent=self.estilos['Normal'],
            backColor=self.colores['fondo_covid'],
            borderColor=self.colores['covid_critico'],
            borderWidth=2,
            borderPadding=20,   # Aumentado padding
            spaceBefore=25,     # Aumentado espaciado antes
            spaceAfter=20,      # Aumentado espaciado después
            leftIndent=15,      # Añadido margen izquierdo
            rightIndent=15      # Añadido margen derecho
        )
        
        elementos.append(Paragraph(disclaimer, disclaimer_style))
        elementos.append(Spacer(1, 15))  # Espaciado entre disclaimer e info adicional
        
        # Información adicional con estilo médico y mejor espaciado
        info_adicional = f"""
        <para align="center" spaceBefore="10" spaceAfter="10">
        <font size="9" color="#1E88E5">
        <b>{t('reportes.proyecto_academico')}</b><br/><br/>
        {t('reportes.no_uso_clinico')}
        </font>
        </para>
        """
        
        info_style = ParagraphStyle(
            'InfoAdicional',
            parent=self.estilos['Normal'],
            backColor=self.colores['fondo_info'],
            borderColor=self.colores['primario'],
            borderWidth=1,
            borderPadding=15,   # Aumentado padding
            spaceAfter=15,      # Aumentado espaciado después
            leftIndent=15,      # Añadido margen izquierdo
            rightIndent=15      # Añadido margen derecho
        )
        
        elementos.append(Paragraph(info_adicional, info_style))
    
    def generar_reporte_analisis_individual(self, resultado_analisis: Dict[str, Any]) -> bytes:
        """
        Genera reporte PDF para análisis individual con espaciado mejorado
        
        Args:
            resultado_analisis: Resultado del análisis de la radiografía
            
        Returns:
            bytes: Contenido del PDF generado
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=80, leftMargin=80,    # Aumentado márgenes
                               topMargin=80, bottomMargin=30)    # Aumentado márgenes
        
        elementos = []
        
        # Encabezado con espaciado mejorado
        self._crear_encabezado(
            elementos,
            t('reportes.analisis_individual.titulo'),
            t('reportes.analisis_individual.subtitulo')
        )
        
        # Resumen ejecutivo con mejor espaciado
        elementos.append(Paragraph(t('reportes.resumen_ejecutivo'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 10))  # Espaciado después del título de sección
        
        # Información del modelo con espaciado mejorado
        modelo_usado = resultado_analisis.get('modelo_usado', 'Custom_CNN')
        clase_predicha = resultado_analisis.get('clase_predicha', 'N/A')
        confianza = resultado_analisis.get('confianza', 0)
        
        resumen_texto = f"""
        <para spaceBefore="5" spaceAfter="5">
        <b>{t('reportes.modelo_utilizado')}:</b> {modelo_usado}<br/>
        <b>{t('reportes.diagnostico_predicho')}:</b> {clase_predicha}<br/>
        <b>{t('reportes.nivel_confianza')}:</b> {confianza:.1f}%<br/>
        <b>{t('reportes.fecha_analisis')}:</b> {self._obtener_fecha_localizada()}
        </para>
        """
        
        elementos.append(Paragraph(resumen_texto, self.estilos['TextoPrincipal']))
        elementos.append(Spacer(1, 25))  # Mayor espaciado entre secciones
        
        # Interpretación del resultado con espaciado mejorado
        elementos.append(Paragraph(t('reportes.interpretacion_resultado'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título de sección
        
        if 'COVID' in clase_predicha.upper():
            interpretacion = f"""
            <para align="center" spaceBefore="10" spaceAfter="10">
            <font color="#D32F2F" size="14">
            <b>🚨 {t('reportes.covid_detectado')} 🚨</b><br/><br/>
            <font color="#1E88E5" size="12">
            {t('reportes.covid_interpretacion')}
            </font>
            </font>
            </para>
            """
            estilo_usado = self.estilos['AlertaCritica']
        elif t('normal').upper() in clase_predicha.upper():
            interpretacion = f"""
            <para align="center" spaceBefore="10" spaceAfter="10">
            <font color="#388E3C" size="13">
            <b>✅ {t('reportes.radiografia_normal')} ✅</b><br/><br/>
            <font color="#1E88E5" size="11">
            {t('reportes.normal_interpretacion')}
            </font>
            </font>
            </para>
            """
            estilo_usado = self.estilos['ResultadoNormal']
        else:
            interpretacion = f"""
            <para align="center" spaceBefore="10" spaceAfter="10">
            <font color="#F57C00" size="13">
            <b>⚠️ {t('reportes.anomalia_detectada')} ⚠️</b><br/><br/>
            <font color="#1E88E5" size="11">
            {t('reportes.anomalia_interpretacion')}
            </font>
            </font>
            </para>
            """
            estilo_usado = self.estilos['Advertencia']
        
        elementos.append(Paragraph(interpretacion, estilo_usado))
        elementos.append(Spacer(1, 30))  # Mayor espaciado después del resultado
        
        # Probabilidades detalladas con tabla mejorada
        elementos.append(Paragraph(t('reportes.probabilidades_detalladas'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        probabilidades = resultado_analisis.get('probabilidades', [])
        clases = [t('covid19'), t('opacidad_pulmonar'), t('normal'), t('neumonia_viral')]
        
        if probabilidades is not None and len(probabilidades) == 4:
            datos_tabla = [[t('reportes.clase'), t('reportes.probabilidad')]]
            
            for i, (clase, prob) in enumerate(zip(clases, probabilidades)):
                marcador = "🎯 " if i == np.argmax(probabilidades) else ""
                datos_tabla.append([f"{marcador}{clase}", f"{prob*100:.1f}%"])
            
            tabla = Table(datos_tabla, colWidths=[4*inch, 2*inch])
            tabla.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colores['primario']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 13),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 15),    # Aumentado padding
                ('TOPPADDING', (0, 0), (-1, 0), 15),       # Aumentado padding
                ('BACKGROUND', (0, 1), (-1, -1), self.colores['fondo']),
                ('FONTSIZE', (0, 1), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 12),   # Aumentado padding
                ('TOPPADDING', (0, 1), (-1, -1), 12),      # Aumentado padding
                ('LEFTPADDING', (0, 0), (-1, -1), 15),     # Añadido padding lateral
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),    # Añadido padding lateral
                ('GRID', (0, 0), (-1, -1), 1.5, self.colores['primario']),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Destacar fila con mayor probabilidad con mejor espaciado
                ('BACKGROUND', (0, 1), (-1, 1), self.colores['fondo_info']),
                ('TEXTCOLOR', (0, 1), (-1, 1), self.colores['primario']),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold')
            ]))
            
            elementos.append(tabla)
            elementos.append(Spacer(1, 30))  # Mayor espaciado después de la tabla
        
        # Recomendaciones médicas con mejor espaciado
        elementos.append(Paragraph(t('reportes.recomendaciones_medicas'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        # Obtener recomendaciones basadas en el resultado
        recomendaciones = self._obtener_recomendaciones_para_pdf(clase_predicha, confianza)
        
        for i, recomendacion in enumerate(recomendaciones):
            elementos.append(Paragraph(f"• {recomendacion}", self.estilos['TextoPrincipal']))
            if i < len(recomendaciones) - 1:  # Espaciado entre recomendaciones
                elementos.append(Spacer(1, 8))
        
        elementos.append(Spacer(1, 30))  # Mayor espaciado después de recomendaciones
        
        # Limitaciones con mejor espaciado
        elementos.append(Paragraph(t('reportes.limitaciones'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        limitaciones_texto = f"""
        <para spaceBefore="5" spaceAfter="8">
        • {t('reportes.limitacion_ia')}<br/><br/>
        • {t('reportes.limitacion_contexto')}<br/><br/>
        • {t('reportes.limitacion_validacion')}<br/><br/>
        • {t('reportes.limitacion_especialista')}
        </para>
        """
        
        elementos.append(Paragraph(limitaciones_texto, self.estilos['TextoPrincipal']))
        
        # Pie de página con espaciado mejorado
        self._crear_pie_pagina(elementos)
        
        # Construir PDF
        doc.build(elementos)
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
    
    def generar_reporte_eda(self, dataset: pd.DataFrame, visualizaciones: Dict[str, Any]) -> bytes:
        """
        Genera reporte PDF para análisis exploratorio con espaciado mejorado
        
        Args:
            dataset: Dataset analizado
            visualizaciones: Diccionario con visualizaciones generadas
            
        Returns:
            bytes: Contenido del PDF generado
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=80, leftMargin=80,    # Aumentado márgenes
                               topMargin=80, bottomMargin=30)    # Aumentado márgenes
        
        elementos = []
        
        # Encabezado con espaciado mejorado
        self._crear_encabezado(
            elementos,
            t('reportes.eda.titulo'),
            t('reportes.eda.subtitulo')
        )
        
        # Resumen del dataset con mejor espaciado
        elementos.append(Paragraph(t('reportes.eda.resumen_dataset'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        # Estadísticas generales con espaciado mejorado
        total_imagenes = len(dataset)
        total_clases = dataset['clase_diagnostico'].nunique()
        distribucion = dataset['clase_diagnostico'].value_counts()
        
        resumen_dataset = f"""
        <para spaceBefore="5" spaceAfter="8">
        <b>{t('reportes.total_imagenes')}:</b> {total_imagenes:,}<br/><br/>
        <b>{t('reportes.total_clases')}:</b> {total_clases}<br/><br/>
        <b>{t('reportes.clase_dominante')}:</b> {distribucion.index[0]} ({(distribucion.iloc[0]/total_imagenes*100):.1f}%)<br/><br/>
        <b>{t('reportes.periodo_analisis')}:</b> {self._obtener_fecha_localizada()}
        </para>
        """
        
        elementos.append(Paragraph(resumen_dataset, self.estilos['TextoPrincipal']))
        elementos.append(Spacer(1, 30))  # Mayor espaciado entre secciones
        
        # Distribución por clase con tabla mejorada
        elementos.append(Paragraph(t('reportes.eda.distribucion_clases'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        datos_distribucion = [[t('reportes.clase'), t('reportes.cantidad'), t('reportes.porcentaje')]]
        
        for clase, cantidad in distribucion.items():
            porcentaje = (cantidad / total_imagenes) * 100
            datos_distribucion.append([clase, str(cantidad), f"{porcentaje:.1f}%"])
        
        tabla_distribucion = Table(datos_distribucion, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        tabla_distribucion.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colores['primario']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 15),    # Aumentado padding
            ('TOPPADDING', (0, 0), (-1, 0), 15),       # Aumentado padding
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 12),   # Aumentado padding
            ('TOPPADDING', (0, 1), (-1, -1), 12),      # Aumentado padding
            ('LEFTPADDING', (0, 0), (-1, -1), 12),     # Añadido padding lateral
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),    # Añadido padding lateral
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elementos.append(tabla_distribucion)
        elementos.append(Spacer(1, 30))  # Mayor espaciado después de la tabla
        
        # Estadísticas descriptivas con mejor espaciado
        elementos.append(Paragraph(t('reportes.eda.estadisticas_descriptivas'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        variables_numericas = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if variables_numericas:
            descripcion = dataset[variables_numericas].describe()
            
            # Crear tabla con estadísticas principales y mejor espaciado
            stats_data = [[t('reportes.variable'), t('reportes.media'), t('reportes.std'), t('reportes.min'), t('reportes.max')]]
            
            for var in variables_numericas[:5]:  # Mostrar solo las primeras 5 variables
                if var in descripcion.columns:
                    stats_data.append([
                        var.replace('_', ' ').title(),
                        f"{descripcion.loc['mean', var]:.3f}",
                        f"{descripcion.loc['std', var]:.3f}",
                        f"{descripcion.loc['min', var]:.3f}",
                        f"{descripcion.loc['max', var]:.3f}"
                    ])
            
            tabla_stats = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            tabla_stats.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colores['secundario']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),   # Aumentado padding
                ('TOPPADDING', (0, 0), (-1, -1), 10),      # Aumentado padding
                ('LEFTPADDING', (0, 0), (-1, -1), 8),      # Añadido padding lateral
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),     # Añadido padding lateral
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elementos.append(tabla_stats)
            elementos.append(Spacer(1, 30))  # Mayor espaciado después de la tabla
        
        # Insights principales con mejor espaciado
        elementos.append(Paragraph(t('reportes.eda.insights_principales'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        insights = self._generar_insights_eda(dataset)
        for i, insight in enumerate(insights):
            elementos.append(Paragraph(f"• {insight}", self.estilos['TextoPrincipal']))
            if i < len(insights) - 1:  # Espaciado entre insights
                elementos.append(Spacer(1, 8))
        
        elementos.append(Spacer(1, 30))  # Mayor espaciado entre secciones
        
        # Recomendaciones para modelado con mejor espaciado
        elementos.append(Paragraph(t('reportes.eda.recomendaciones_modelado'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        recomendaciones_modelado = [
            t('reportes.eda.recomendacion_1'),
            t('reportes.eda.recomendacion_2'),
            t('reportes.eda.recomendacion_3'),
            t('reportes.eda.recomendacion_4')
        ]
        
        for i, recomendacion in enumerate(recomendaciones_modelado):
            elementos.append(Paragraph(f"• {recomendacion}", self.estilos['TextoPrincipal']))
            if i < len(recomendaciones_modelado) - 1:  # Espaciado entre recomendaciones
                elementos.append(Spacer(1, 8))
        
        # Pie de página con espaciado mejorado
        self._crear_pie_pagina(elementos)
        
        # Construir PDF
        doc.build(elementos)
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
    
    def generar_reporte_comparacion_modelos(self, comparacion_resultados: Dict[str, Any]) -> bytes:
        """
        Genera reporte PDF para comparación de modelos con espaciado mejorado
        
        Args:
            comparacion_resultados: Resultados de la comparación de modelos
            
        Returns:
            bytes: Contenido del PDF generado
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=80, leftMargin=80,    # Aumentado márgenes
                               topMargin=80, bottomMargin=30)    # Aumentado márgenes
        
        elementos = []
        
        # Encabezado con espaciado mejorado
        self._crear_encabezado(
            elementos,
            t('reportes.comparacion.titulo'),
            t('reportes.comparacion.subtitulo')
        )
        
        # Ranking de modelos con mejor espaciado
        elementos.append(Paragraph(t('reportes.comparacion.ranking_modelos'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        ranking = comparacion_resultados.get('ranking_modelos', [])
        
        if ranking:
            # Destacar el mejor modelo con espaciado mejorado
            mejor_modelo = ranking[0]
            
            destacado_texto = f"""
            <para spaceBefore="10" spaceAfter="15">
            <font color="#008000">
            <b>🏆 {t('reportes.mejor_modelo')}:</b> {mejor_modelo['modelo']}<br/><br/>
            <b>📊 Score Global:</b> {mejor_modelo['score_global']:.2f}%<br/><br/>
            <b>🎯 Accuracy:</b> {mejor_modelo['accuracy']:.3f}<br/><br/>
            <b>📈 MCC:</b> {mejor_modelo['mcc']:.3f}
            </font>
            </para>
            """
            
            elementos.append(Paragraph(destacado_texto, self.estilos['InfoMedica']))
            elementos.append(Spacer(1, 20))  # Mayor espaciado después del destacado
            
            # Tabla de ranking completo con mejor espaciado
            datos_ranking = [[t('reportes.posicion'), t('reportes.modelo'), t('reportes.score_global'), 
                            t('reportes.accuracy'), t('reportes.f1_score'), t('reportes.mcc')]]
            
            for modelo_info in ranking:
                datos_ranking.append([
                    str(modelo_info['posicion']),
                    modelo_info['modelo'],
                    f"{modelo_info['score_global']:.1f}%",
                    f"{modelo_info['accuracy']:.3f}",
                    f"{modelo_info['f1_macro']:.3f}",
                    f"{modelo_info['mcc']:.3f}"
                ])
            
            tabla_ranking = Table(datos_ranking, colWidths=[0.8*inch, 2*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
            tabla_ranking.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colores['primario']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),   # Aumentado padding
                ('TOPPADDING', (0, 0), (-1, -1), 10),      # Aumentado padding
                ('LEFTPADDING', (0, 0), (-1, -1), 8),      # Añadido padding lateral
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),     # Añadido padding lateral
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                # Destacar el primer lugar con mejor espaciado
                ('BACKGROUND', (0, 1), (-1, 1), HexColor('#E8F5E8')),
                ('TEXTCOLOR', (0, 1), (-1, 1), HexColor('#006400')),
                ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold')
            ]))
            
            elementos.append(tabla_ranking)
            elementos.append(Spacer(1, 30))  # Mayor espaciado después de la tabla
        
        # Análisis estadístico con mejor espaciado
        elementos.append(Paragraph(t('reportes.comparacion.analisis_estadistico'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        tests_mcnemar = comparacion_resultados.get('tests_mcnemar', {})
        
        if tests_mcnemar:
            significativos = sum(1 for resultado in tests_mcnemar.values() if resultado.get('significativo', False))
            total_tests = len(tests_mcnemar)
            
            analisis_texto = f"""
            <para spaceBefore="5" spaceAfter="10">
            <b>{t('reportes.tests_realizados')}:</b> {total_tests}<br/><br/>
            <b>{t('reportes.diferencias_significativas')}:</b> {significativos}/{total_tests}<br/><br/>
            <b>{t('reportes.criterio_significancia')}:</b> p < 0.05 ({t('reportes.test_mcnemar')})<br/>
            </para>
            """
            
            elementos.append(Paragraph(analisis_texto, self.estilos['TextoPrincipal']))
            elementos.append(Spacer(1, 25))  # Mayor espaciado entre secciones
        
        # Recomendaciones finales con mejor espaciado
        elementos.append(Paragraph(t('reportes.comparacion.recomendaciones_finales'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        if ranking:
            mejor_modelo = ranking[0]['modelo']
            segundo_modelo = ranking[1]['modelo'] if len(ranking) > 1 else "N/A"
            
            recomendaciones_finales = [
                f"{t('reportes.recomendacion_produccion')} {mejor_modelo}",
                f"{t('reportes.recomendacion_backup')} {segundo_modelo}",
                t('reportes.recomendacion_ensemble'),
                t('reportes.recomendacion_monitoreo'),
                t('reportes.recomendacion_validacion')
            ]
            
            for i, recomendacion in enumerate(recomendaciones_finales):
                elementos.append(Paragraph(f"• {recomendacion}", self.estilos['TextoPrincipal']))
                if i < len(recomendaciones_finales) - 1:  # Espaciado entre recomendaciones
                    elementos.append(Spacer(1, 8))
        
        elementos.append(Spacer(1, 30))  # Mayor espaciado entre secciones
        
        # Limitaciones del estudio con mejor espaciado
        elementos.append(Paragraph(t('reportes.comparacion.limitaciones'), self.estilos['Seccion']))
        elementos.append(Spacer(1, 15))  # Espaciado después del título
        
        limitaciones = [
            t('reportes.limitacion_datos_sinteticos'),
            t('reportes.limitacion_evaluacion_laboratorio'),
            t('reportes.limitacion_contexto_clinico'),
            t('reportes.limitacion_validacion_externa')
        ]
        
        for i, limitacion in enumerate(limitaciones):
            elementos.append(Paragraph(f"• {limitacion}", self.estilos['TextoPrincipal']))
            if i < len(limitaciones) - 1:  # Espaciado entre limitaciones
                elementos.append(Spacer(1, 8))
        
        # Pie de página con espaciado mejorado
        self._crear_pie_pagina(elementos)
        
        # Construir PDF
        doc.build(elementos)
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
    
    def _obtener_recomendaciones_para_pdf(self, clase_predicha: str, confianza: float) -> List[str]:
        """Obtiene recomendaciones médicas específicas para incluir en el PDF"""
        if 'COVID' in clase_predicha.upper():
            if confianza >= 70:
                return [
                    t('reportes.recomendaciones.covid_alta_confianza_1'),
                    t('reportes.recomendaciones.covid_alta_confianza_2'),
                    t('reportes.recomendaciones.covid_alta_confianza_3'),
                    t('reportes.recomendaciones.covid_alta_confianza_4')
                ]
            else:
                return [
                    t('reportes.recomendaciones.covid_baja_confianza_1'),
                    t('reportes.recomendaciones.covid_baja_confianza_2'),
                    t('reportes.recomendaciones.covid_baja_confianza_3')
                ]
        elif t('normal').upper() in clase_predicha.upper():
            return [
                t('reportes.recomendaciones.normal_1'),
                t('reportes.recomendaciones.normal_2'),
                t('reportes.recomendaciones.normal_3')
            ]
        else:
            return [
                t('reportes.recomendaciones.anomalia_1'),
                t('reportes.recomendaciones.anomalia_2'),
                t('reportes.recomendaciones.anomalia_3'),
                t('reportes.recomendaciones.anomalia_4')
            ]
    
    def _generar_insights_eda(self, dataset: pd.DataFrame) -> List[str]:
        """Genera insights automáticos del análisis EDA"""
        insights = []
        
        # Balance de clases
        distribucion = dataset['clase_diagnostico'].value_counts()
        balance_ratio = distribucion.min() / distribucion.max()
        
        if balance_ratio > 0.7:
            insights.append(t('reportes.insights.dataset_balanceado'))
        elif balance_ratio > 0.3:
            insights.append(t('reportes.insights.dataset_moderadamente_desbalanceado'))
        else:
            insights.append(t('reportes.insights.dataset_muy_desbalanceado'))
        
        # Calidad de datos
        valores_faltantes_pct = (dataset.isnull().sum().sum() / dataset.size) * 100
        if valores_faltantes_pct == 0:
            insights.append(t('reportes.insights.datos_completos'))
        elif valores_faltantes_pct < 5:
            insights.append(f"{t('reportes.insights.buena_calidad')} ({valores_faltantes_pct:.1f}%)")
        else:
            insights.append(f"{t('reportes.insights.requiere_limpieza')} ({valores_faltantes_pct:.1f}%)")
        
        # Diversidad etaria
        if 'edad_paciente' in dataset.columns:
            edad_std = dataset['edad_paciente'].std()
            if edad_std > 15:
                insights.append(t('reportes.insights.amplio_rango_etario'))
            else:
                insights.append(t('reportes.insights.rango_etario_limitado'))
        
        return insights
    
    def crear_nombre_archivo(self, tipo_reporte: str, extension: str = "pdf") -> str:
        """
        Crea nombre de archivo localizado según el idioma
        
        Args:
            tipo_reporte: Tipo de reporte ('analisis', 'eda', 'comparacion')
            extension: Extensión del archivo
            
        Returns:
            str: Nombre del archivo localizado
        """
        fecha = datetime.now().strftime("%Y%m%d")
        idioma = gestor_multilenguaje.obtener_idioma_actual() if MULTILENGUAJE_DISPONIBLE else 'es'
        
        nombres_tipos = {
            'analisis': {
                'es': 'reporte_analisis_radiografia',
                'en': 'xray_analysis_report',
                'fr': 'rapport_analyse_radiographie'
            },
            'eda': {
                'es': 'reporte_analisis_exploratorio',
                'en': 'exploratory_data_analysis_report',
                'fr': 'rapport_analyse_exploratoire'
            },
            'comparacion': {
                'es': 'reporte_comparacion_modelos',
                'en': 'model_comparison_report',
                'fr': 'rapport_comparaison_modeles'
            }
        }
        
        nombre_base = nombres_tipos.get(tipo_reporte, {}).get(idioma, f"reporte_{tipo_reporte}")
        
        return f"{nombre_base}_{fecha}_{idioma}.{extension}"

# Instancia global del generador mejorado
generador_reportes = GeneradorReportesPDF()