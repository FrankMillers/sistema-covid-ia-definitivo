"""
Módulo de Análisis Exploratorio de Datos (EDA) para COVID-19
Autor: Sistema IA COVID-19
Descripción: Análisis estadístico completo de radiografías COVID-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
import cv2
import os
from typing import Dict, List, Tuple, Any
import io
import base64

class AnalizadorEDACovid:
    """Clase para análisis exploratorio de datos de COVID-19 en radiografías"""
    
    def __init__(self):
        """Inicializa el analizador EDA"""
        self.datos_metadatos = None
        self.estadisticos_descriptivos = None
        self.correlaciones_caracteristicas = None
        self.configurar_estilo_visualizaciones()
        
    def configurar_estilo_visualizaciones(self):
        """Configura el estilo general para las visualizaciones"""
        # Configuración matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configuración plotly
        self.colores_covid = {
            'COVID-19': '#FF6B6B',
            'Normal': '#4ECDC4', 
            'Opacidad Pulmonar': '#45B7D1',
            'Neumonía Viral': '#FFA726'
        }
        
    def generar_dataset_simulado(self, tamaño_muestra: int = 1000) -> pd.DataFrame:
        """
        Genera un dataset simulado con características realistas de radiografías COVID-19
        
        Args:
            tamaño_muestra (int): Número de muestras a generar
            
        Returns:
            pd.DataFrame: Dataset con metadatos de radiografías
        """
        np.random.seed(42)  # Para reproducibilidad
        
        # Distribución de clases realista
        clases = ['Normal', 'COVID-19', 'Opacidad Pulmonar', 'Neumonía Viral']
        proporcion_clases = [0.4, 0.25, 0.20, 0.15]  # Proporción realista
        
        datos = []
        
        for i in range(tamaño_muestra):
            # Asignar clase basada en proporción
            clase = np.random.choice(clases, p=proporcion_clases)
            
            # Generar características basadas en la clase
            if clase == 'COVID-19':
                opacidad_media = np.random.normal(0.65, 0.15)
                contraste_medio = np.random.normal(0.45, 0.12)
                textura_rugosidad = np.random.normal(0.7, 0.1)
                densidad_pulmonar = np.random.normal(0.6, 0.1)
                patron_bilateral = np.random.choice([0, 1], p=[0.3, 0.7])
                
            elif clase == 'Normal':
                opacidad_media = np.random.normal(0.25, 0.1)
                contraste_medio = np.random.normal(0.7, 0.1)
                textura_rugosidad = np.random.normal(0.3, 0.1)
                densidad_pulmonar = np.random.normal(0.3, 0.1)
                patron_bilateral = np.random.choice([0, 1], p=[0.9, 0.1])
                
            elif clase == 'Opacidad Pulmonar':
                opacidad_media = np.random.normal(0.55, 0.12)
                contraste_medio = np.random.normal(0.5, 0.1)
                textura_rugosidad = np.random.normal(0.6, 0.12)
                densidad_pulmonar = np.random.normal(0.55, 0.1)
                patron_bilateral = np.random.choice([0, 1], p=[0.6, 0.4])
                
            else:  # Neumonía Viral
                opacidad_media = np.random.normal(0.6, 0.1)
                contraste_medio = np.random.normal(0.4, 0.1)
                textura_rugosidad = np.random.normal(0.65, 0.1)
                densidad_pulmonar = np.random.normal(0.65, 0.1)
                patron_bilateral = np.random.choice([0, 1], p=[0.5, 0.5])
            
            # Generar metadatos adicionales
            edad = np.random.normal(55, 20)
            edad = max(18, min(90, edad))  # Limitar rango de edad
            
            # Género
            genero = np.random.choice(['Masculino', 'Femenino'], p=[0.52, 0.48])
            
            # Características técnicas de la imagen
            resolucion_x = np.random.choice([256, 512, 1024], p=[0.3, 0.5, 0.2])
            resolucion_y = resolucion_x  # Mantener aspecto cuadrado
            bits_profundidad = np.random.choice([8, 16], p=[0.7, 0.3])
            
            # Características calculadas de la imagen
            entropia_imagen = np.random.normal(7.2, 0.8)
            energia_imagen = np.random.normal(0.15, 0.05)
            homogeneidad = np.random.normal(0.3, 0.1)
            
            # Normalizar valores entre 0 y 1
            opacidad_media = np.clip(opacidad_media, 0, 1)
            contraste_medio = np.clip(contraste_medio, 0, 1)
            textura_rugosidad = np.clip(textura_rugosidad, 0, 1)
            densidad_pulmonar = np.clip(densidad_pulmonar, 0, 1)
            homogeneidad = np.clip(homogeneidad, 0, 1)
            energia_imagen = np.clip(energia_imagen, 0, 1)
            
            datos.append({
                'id_imagen': f'IMG_{i:04d}',
                'clase_diagnostico': clase,
                'edad_paciente': int(edad),
                'genero_paciente': genero,
                'opacidad_media': round(opacidad_media, 3),
                'contraste_medio': round(contraste_medio, 3),
                'textura_rugosidad': round(textura_rugosidad, 3),
                'densidad_pulmonar': round(densidad_pulmonar, 3),
                'patron_bilateral': patron_bilateral,
                'resolucion_x': resolucion_x,
                'resolucion_y': resolucion_y,
                'bits_profundidad': bits_profundidad,
                'entropia_imagen': round(entropia_imagen, 3),
                'energia_imagen': round(energia_imagen, 3),
                'homogeneidad': round(homogeneidad, 3)
            })
        
        return pd.DataFrame(datos)
    
    def calcular_estadisticos_descriptivos(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula estadísticos descriptivos completos del dataset
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            Dict[str, Any]: Diccionario con estadísticos descriptivos
        """
        estadisticos = {
            'resumen_general': {},
            'por_clase': {},
            'correlaciones': {},
            'valores_faltantes': {}
        }
        
        # Resumen general
        estadisticos['resumen_general'] = {
            'total_imagenes': len(dataset),
            'total_variables': len(dataset.columns),
            'distribucion_clases': dataset['clase_diagnostico'].value_counts().to_dict(),
            'proporcion_clases': (dataset['clase_diagnostico'].value_counts(normalize=True) * 100).round(2).to_dict()
        }
        
        # Estadísticos por clase
        variables_numericas = dataset.select_dtypes(include=[np.number]).columns
        
        for clase in dataset['clase_diagnostico'].unique():
            datos_clase = dataset[dataset['clase_diagnostico'] == clase]
            estadisticos['por_clase'][clase] = {
                'cantidad': len(datos_clase),
                'edad_promedio': datos_clase['edad_paciente'].mean(),
                'edad_std': datos_clase['edad_paciente'].std(),
                'opacidad_promedio': datos_clase['opacidad_media'].mean(),
                'contraste_promedio': datos_clase['contraste_medio'].mean(),
                'densidad_promedio': datos_clase['densidad_pulmonar'].mean()
            }
        
        # Matriz de correlaciones
        correlaciones = dataset[variables_numericas].corr()
        estadisticos['correlaciones'] = correlaciones
        
        # Valores faltantes
        estadisticos['valores_faltantes'] = {
            'total_faltantes': dataset.isnull().sum().sum(),
            'por_columna': dataset.isnull().sum().to_dict(),
            'porcentaje_faltantes': (dataset.isnull().sum() / len(dataset) * 100).round(2).to_dict()
        }
        
        return estadisticos
    
    def crear_mapa_calor_correlaciones(self, dataset: pd.DataFrame) -> go.Figure:
        """
        Crea mapa de calor de correlaciones entre características de radiografías
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            go.Figure: Figura plotly con mapa de calor
        """
        # Seleccionar solo variables numéricas relevantes para radiografías
        variables_interes = [
            'opacidad_media', 'contraste_medio', 'textura_rugosidad',
            'densidad_pulmonar', 'entropia_imagen', 'energia_imagen', 'homogeneidad'
        ]
        
        # Filtrar variables que existan en el dataset
        variables_disponibles = [var for var in variables_interes if var in dataset.columns]
        
        # Calcular matriz de correlaciones
        matriz_correlacion = dataset[variables_disponibles].corr()
        
        # Crear mapa de calor
        fig = go.Figure(data=go.Heatmap(
            z=matriz_correlacion.values,
            x=[self._traducir_variable(var) for var in matriz_correlacion.columns],
            y=[self._traducir_variable(var) for var in matriz_correlacion.index],
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(matriz_correlacion.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title={
                'text': '🔥 Mapa de Calor - Correlaciones entre Características de Radiografías',
                'x': 0.5,
                'font': {'size': 16, 'color': '#2E86AB'}
            },
            xaxis_title="Características de Imagen",
            yaxis_title="Características de Imagen",
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def crear_histogramas_caracteristicas(self, dataset: pd.DataFrame) -> go.Figure:
        """
        Crea histogramas de las principales características por clase
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            go.Figure: Figura plotly con subplots de histogramas
        """
        # Variables principales para histogramas
        variables = ['opacidad_media', 'contraste_medio', 'textura_rugosidad', 'densidad_pulmonar']
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[self._traducir_variable(var) for var in variables],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        posiciones = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, variable in enumerate(variables):
            row, col = posiciones[i]
            
            for clase in dataset['clase_diagnostico'].unique():
                datos_clase = dataset[dataset['clase_diagnostico'] == clase][variable]
                
                fig.add_trace(
                    go.Histogram(
                        x=datos_clase,
                        name=clase,
                        opacity=0.7,
                        nbinsx=25,
                        legendgroup=clase,
                        showlegend=(i == 0),  # Solo mostrar leyenda en el primer subplot
                        marker_color=self.colores_covid.get(clase, '#888888')
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title={
                'text': '📊 Distribución de Características por Clase Diagnóstica',
                'x': 0.5,
                'font': {'size': 16, 'color': '#2E86AB'}
            },
            height=700,
            showlegend=True,
            legend=dict(x=1.02, y=1),
            barmode='overlay'
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Valor de la Característica")
        fig.update_yaxes(title_text="Frecuencia")
        
        return fig
    
    def crear_boxplots_comparativos(self, dataset: pd.DataFrame) -> go.Figure:
        """
        Crea boxplots comparativos de características por clase
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            go.Figure: Figura plotly con boxplots
        """
        variables = ['opacidad_media', 'contraste_medio', 'densidad_pulmonar', 'edad_paciente']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[self._traducir_variable(var) for var in variables],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        posiciones = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, variable in enumerate(variables):
            row, col = posiciones[i]
            
            for clase in dataset['clase_diagnostico'].unique():
                datos_clase = dataset[dataset['clase_diagnostico'] == clase][variable]
                
                fig.add_trace(
                    go.Box(
                        y=datos_clase,
                        name=clase,
                        legendgroup=clase,
                        showlegend=(i == 0),
                        marker_color=self.colores_covid.get(clase, '#888888'),
                        boxmean='sd'  # Mostrar media y desviación estándar
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title={
                'text': '📦 Análisis de Distribución por Boxplots - Comparación entre Clases',
                'x': 0.5,
                'font': {'size': 16, 'color': '#2E86AB'}
            },
            height=700,
            showlegend=True,
            legend=dict(x=1.02, y=1)
        )
        
        return fig
    
    def crear_scatter_plot_multivariable(self, dataset: pd.DataFrame) -> go.Figure:
        """
        Crea scatter plot multivariable para explorar relaciones
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            go.Figure: Figura plotly con scatter plot
        """
        fig = go.Figure()
        
        for clase in dataset['clase_diagnostico'].unique():
            datos_clase = dataset[dataset['clase_diagnostico'] == clase]
            
            fig.add_trace(
                go.Scatter(
                    x=datos_clase['opacidad_media'],
                    y=datos_clase['contraste_medio'],
                    mode='markers',
                    name=clase,
                    marker=dict(
                        size=datos_clase['densidad_pulmonar'] * 20,  # Tamaño basado en densidad
                        color=self.colores_covid.get(clase, '#888888'),
                        opacity=0.7,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=[f'Edad: {edad}<br>Textura: {textura:.3f}' 
                          for edad, textura in zip(datos_clase['edad_paciente'], datos_clase['textura_rugosidad'])],
                    hovertemplate='%{text}<br>Opacidad: %{x}<br>Contraste: %{y}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title={
                'text': '🎯 Análisis Multivariable: Opacidad vs Contraste<br><sub>Tamaño del punto = Densidad Pulmonar</sub>',
                'x': 0.5,
                'font': {'size': 16, 'color': '#2E86AB'}
            },
            xaxis_title="Opacidad Media de la Imagen",
            yaxis_title="Contraste Medio de la Imagen",
            height=600,
            width=800
        )
        
        return fig
    
    def _traducir_variable(self, variable: str) -> str:
        """Traduce nombres de variables técnicas a español legible"""
        traduccion = {
            'opacidad_media': 'Opacidad Media',
            'contraste_medio': 'Contraste Medio',
            'textura_rugosidad': 'Rugosidad de Textura',
            'densidad_pulmonar': 'Densidad Pulmonar',
            'entropia_imagen': 'Entropía de Imagen',
            'energia_imagen': 'Energía de Imagen', 
            'homogeneidad': 'Homogeneidad',
            'edad_paciente': 'Edad del Paciente'
        }
        return traduccion.get(variable, variable.replace('_', ' ').title())
    
    def generar_reporte_eda_completo(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera reporte EDA completo con todas las visualizaciones y estadísticos
        
        Args:
            dataset (pd.DataFrame): Dataset de radiografías
            
        Returns:
            Dict[str, Any]: Reporte completo con figuras y estadísticos
        """
        reporte = {
            'estadisticos': self.calcular_estadisticos_descriptivos(dataset),
            'visualizaciones': {
                'mapa_calor_correlaciones': self.crear_mapa_calor_correlaciones(dataset),
                'histogramas_caracteristicas': self.crear_histogramas_caracteristicas(dataset),
                'boxplots_comparativos': self.crear_boxplots_comparativos(dataset),
                'scatter_multivariable': self.crear_scatter_plot_multivariable(dataset)
            },
            'dataset': dataset,
            'resumen_ejecutivo': self._generar_resumen_ejecutivo(dataset)
        }
        
        return reporte
    
    def _generar_resumen_ejecutivo(self, dataset: pd.DataFrame) -> Dict[str, str]:
        """Genera resumen ejecutivo del análisis EDA"""
        stats = self.calcular_estadisticos_descriptivos(dataset)
        
        total_imagenes = stats['resumen_general']['total_imagenes']
        clase_dominante = max(stats['resumen_general']['distribucion_clases'], 
                            key=stats['resumen_general']['distribucion_clases'].get)
        
        # Calcular correlación más fuerte
        matriz_corr = dataset[['opacidad_media', 'contraste_medio', 'densidad_pulmonar']].corr()
        np.fill_diagonal(matriz_corr.values, 0)  # Excluir correlación consigo mismo
        max_corr_idx = np.unravel_index(np.argmax(np.abs(matriz_corr.values)), matriz_corr.shape)
        var1, var2 = matriz_corr.index[max_corr_idx[0]], matriz_corr.columns[max_corr_idx[1]]
        max_corr_value = matriz_corr.iloc[max_corr_idx]
        
        resumen = {
            'total_casos': f"📊 Dataset con {total_imagenes:,} radiografías analizadas",
            'distribucion': f"🎯 Clase dominante: {clase_dominante} ({stats['resumen_general']['proporcion_clases'][clase_dominante]:.1f}%)",
            'correlacion_principal': f"🔗 Mayor correlación: {self._traducir_variable(var1)} ↔ {self._traducir_variable(var2)} (r={max_corr_value:.3f})",
            'calidad_datos': f"✅ Datos completos: {100 - (stats['valores_faltantes']['total_faltantes']/total_imagenes*100):.1f}% de completitud"
        }
        
        return resumen

# Instancia global del analizador
analizador_eda = AnalizadorEDACovid()