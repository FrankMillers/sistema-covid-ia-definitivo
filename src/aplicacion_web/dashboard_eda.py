"""
Dashboard EDA para Streamlit - Sistema COVID-19 IA
Autor: Sistema IA COVID-19  
Descripción: Dashboard interactivo para análisis exploratorio de datos con reportes PDF
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Agregar rutas para imports
sys.path.append('src')

try:
    from sistema_multilenguaje.sistema_multilenguaje import t, gestor_multilenguaje
    from analisis_eda.analizador_eda_covid import analizador_eda
    MODULOS_DISPONIBLES = True
except ImportError as e:
    print(f"⚠️ Error importando módulos: {e}")
    MODULOS_DISPONIBLES = False
    
    # Funciones fallback
    def t(clave): return clave

# Importar generador de reportes PDF
try:
    from reportes_pdf.generador_reportes import generador_reportes
    REPORTES_DISPONIBLES = True
except ImportError:
    print("⚠️ Generador de reportes PDF no disponible")
    REPORTES_DISPONIBLES = False

class DashboardEDA:
    """Dashboard interactivo para análisis exploratorio de datos COVID-19"""
    
    def __init__(self):
        """Inicializa el dashboard EDA"""
        self.dataset_cache = None
        self.reporte_cache = None
        
    @st.cache_data
    def cargar_dataset_covid(_self, tamaño_muestra: int = 1000) -> pd.DataFrame:
        """
        Carga o genera dataset COVID-19 para análisis
        
        Args:
            tamaño_muestra (int): Tamaño del dataset a generar
            
        Returns:
            pd.DataFrame: Dataset de radiografías COVID-19
        """
        if not MODULOS_DISPONIBLES:
            # Dataset simple de fallback
            np.random.seed(42)
            clases = ['Normal', 'COVID-19', 'Opacidad Pulmonar', 'Neumonía Viral']
            datos = []
            
            for i in range(tamaño_muestra):
                datos.append({
                    'id_imagen': f'IMG_{i:04d}',
                    'clase_diagnostico': np.random.choice(clases),
                    'edad_paciente': np.random.randint(18, 90),
                    'opacidad_media': np.random.random(),
                    'contraste_medio': np.random.random(),
                    'densidad_pulmonar': np.random.random()
                })
            return pd.DataFrame(datos)
        
        # Usar analizador completo
        return analizador_eda.generar_dataset_simulado(tamaño_muestra)
    
    def mostrar_resumen_ejecutivo(self, dataset: pd.DataFrame):
        """Muestra resumen ejecutivo del dataset"""
        st.subheader(t("eda.resumen_ejecutivo"))
        
        if MODULOS_DISPONIBLES:
            resumen = analizador_eda._generar_resumen_ejecutivo(dataset)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=t("eda.total_radiografias"),
                    value=f"{len(dataset):,}",
                    delta=t("eda.dataset_completo")
                )
                
                # Distribución de clases
                distribucion = dataset['clase_diagnostico'].value_counts()
                clase_dominante = distribucion.index[0]
                porcentaje_dominante = (distribucion.iloc[0] / len(dataset)) * 100
                
                st.metric(
                    label=t("eda.clase_dominante"),
                    value=clase_dominante,
                    delta=f"{porcentaje_dominante:.1f}% {t('eda.del_total')}"
                )
            
            with col2:
                # Calidad de datos
                valores_completos = dataset.notna().all(axis=1).sum()
                porcentaje_completo = (valores_completos / len(dataset)) * 100
                
                st.metric(
                    label=t("eda.completitud_datos"),
                    value=f"{porcentaje_completo:.1f}%",
                    delta=t("eda.sin_valores_faltantes") if porcentaje_completo == 100 else f"{100-porcentaje_completo:.1f}% incompleto"
                )
                
                # Edad promedio
                if 'edad_paciente' in dataset.columns:
                    edad_promedio = dataset['edad_paciente'].mean()
                    st.metric(
                        label=t("eda.edad_promedio"),
                        value=f"{edad_promedio:.1f} {t('eda.años')}",
                        delta=f"{t('eda.rango')}: {dataset['edad_paciente'].min()}-{dataset['edad_paciente'].max()}"
                    )
        else:
            st.info("📊 Dataset básico cargado - funcionalidad limitada sin módulos completos")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Imágenes", len(dataset))
            with col2:
                st.metric("Clases", dataset['clase_diagnostico'].nunique())
            with col3:
                st.metric("Variables", len(dataset.columns))
    
    def mostrar_distribucion_clases(self, dataset: pd.DataFrame):
        """Muestra distribución de clases diagnósticas"""
        st.subheader(t("eda.distribucion_clases"))
        
        # Calcular distribución
        distribucion = dataset['clase_diagnostico'].value_counts()
        porcentajes = (distribucion / len(dataset) * 100).round(1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de pastel
            fig_pie = px.pie(
                values=distribucion.values,
                names=distribucion.index,
                title=t("eda.distribucion_clases"),
                color_discrete_map={
                    'COVID-19': '#FF6B6B',
                    'Normal': '#4ECDC4',
                    'Opacidad Pulmonar': '#45B7D1',
                    'Neumonía Viral': '#FFA726'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Tabla de resumen
            st.markdown(f"### {t('eda.estadisticas_clase')}")
            tabla_resumen = pd.DataFrame({
                'Clase': distribucion.index,
                'Cantidad': distribucion.values,
                'Porcentaje': [f"{p}%" for p in porcentajes]
            })
            st.dataframe(tabla_resumen, use_container_width=True)
            
            # Estadísticas adicionales
            st.markdown(f"### {t('eda.metricas_clave')}")
            st.metric(t("eda.clase_mas_comun"), distribucion.index[0])
            st.metric(t("eda.clase_menos_comun"), distribucion.index[-1])
            
            # Balance del dataset
            balance = (distribucion.max() - distribucion.min()) / distribucion.max() * 100
            balance_label = t("eda.bien_balanceado") if balance < 50 else t("eda.desbalanceado")
            st.metric(t("eda.balance"), f"{balance:.1f}%", balance_label)
    
    def mostrar_estadisticos_descriptivos(self, dataset: pd.DataFrame):
        """Muestra estadísticos descriptivos del dataset"""
        st.subheader(t("eda.estadisticos_descriptivos"))
        
        # Seleccionar variables numéricas
        variables_numericas = dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if not variables_numericas:
            st.warning("⚠️ No se encontraron variables numéricas para analizar")
            return
        
        # Mostrar descripción general
        st.markdown(f"### {t('eda.estadisticos_generales')}")
        descripcion = dataset[variables_numericas].describe()
        st.dataframe(descripcion.round(3), use_container_width=True)
        
        # Estadísticos por clase
        st.markdown(f"### {t('eda.estadisticos_por_clase')}")
        
        if 'clase_diagnostico' in dataset.columns:
            # Crear tabs para cada variable
            variables_interes = ['opacidad_media', 'contraste_medio', 'densidad_pulmonar', 'edad_paciente']
            variables_disponibles = [var for var in variables_interes if var in dataset.columns]
            
            if variables_disponibles:
                tabs = st.tabs([self._traducir_variable(var) for var in variables_disponibles])
                
                for i, variable in enumerate(variables_disponibles):
                    with tabs[i]:
                        # Estadísticos por clase para esta variable
                        stats_por_clase = dataset.groupby('clase_diagnostico')[variable].agg([
                            'count', 'mean', 'std', 'min', 'max'
                        ]).round(3)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.dataframe(stats_por_clase, use_container_width=True)
                        
                        with col2:
                            # Visualización rápida
                            fig_box = px.box(
                                dataset, 
                                x='clase_diagnostico', 
                                y=variable,
                                title=f"{t('eda.distribucion_caracteristicas')} - {self._traducir_variable(variable)}",
                                color='clase_diagnostico',
                                color_discrete_map={
                                    'COVID-19': '#FF6B6B',
                                    'Normal': '#4ECDC4',
                                    'Opacidad Pulmonar': '#45B7D1',
                                    'Neumonía Viral': '#FFA726'
                                }
                            )
                            fig_box.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig_box, use_container_width=True)
        
        # Información adicional de calidad de datos
        st.markdown(f"### {t('eda.calidad_datos')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            valores_faltantes = dataset.isnull().sum().sum()
            st.metric(
                t("eda.valores_faltantes"),
                valores_faltantes,
                f"{(valores_faltantes/dataset.size)*100:.2f}% del total"
            )
        
        with col2:
            duplicados = dataset.duplicated().sum()
            st.metric(
                t("eda.filas_duplicadas"),
                duplicados,
                f"{(duplicados/len(dataset))*100:.2f}% del total"
            )
        
        with col3:
            tipos_datos = dataset.dtypes.value_counts()
            st.metric(
                t("eda.tipos_datos"),
                len(tipos_datos),
                f"{tipos_datos.index[0]} {t('eda.dominante')}"
            )
    
    def mostrar_mapa_calor_correlaciones(self, dataset: pd.DataFrame):
        """Muestra mapa de calor de correlaciones"""
        st.subheader(t("eda.mapa_calor"))
        
        # Seleccionar variables numéricas relevantes
        variables_interes = [
            'opacidad_media', 'contraste_medio', 'textura_rugosidad',
            'densidad_pulmonar', 'edad_paciente'
        ]
        
        # Filtrar variables disponibles
        variables_disponibles = [var for var in variables_interes if var in dataset.columns]
        
        if len(variables_disponibles) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para calcular correlaciones")
            return
        
        # Calcular matriz de correlaciones
        matriz_correlacion = dataset[variables_disponibles].corr()
        
        if MODULOS_DISPONIBLES:
            # Usar visualización avanzada
            fig = analizador_eda.crear_mapa_calor_correlaciones(dataset)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Visualización básica con plotly express
            fig = px.imshow(
                matriz_correlacion,
                text_auto=True,
                aspect="auto",
                title=t("eda.mapa_calor"),
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar correlaciones más fuertes
        st.markdown(f"### {t('eda.correlaciones_significativas')}")
        
        # Extraer correlaciones (excluyendo diagonal)
        mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))
        correlaciones_flat = matriz_correlacion.mask(mask).stack().reset_index()
        correlaciones_flat.columns = ['Variable 1', 'Variable 2', 'Correlación']
        correlaciones_flat = correlaciones_flat.sort_values('Correlación', key=abs, ascending=False)
        
        # Mostrar top 5 correlaciones
        top_correlaciones = correlaciones_flat.head(5)
        top_correlaciones['Variable 1'] = top_correlaciones['Variable 1'].apply(self._traducir_variable)
        top_correlaciones['Variable 2'] = top_correlaciones['Variable 2'].apply(self._traducir_variable)
        top_correlaciones['Correlación'] = top_correlaciones['Correlación'].round(3)
        
        st.dataframe(top_correlaciones, use_container_width=True)
        
        # Interpretación automática
        if not top_correlaciones.empty:
            correlacion_maxima = top_correlaciones.iloc[0]
            interpretacion = self._interpretar_correlacion(correlacion_maxima['Correlación'])
            
            st.info(f"""
            🎯 **Correlación más fuerte**: {correlacion_maxima['Variable 1']} ↔ {correlacion_maxima['Variable 2']}
            
            📊 **Valor**: {correlacion_maxima['Correlación']:.3f} ({interpretacion})
            """)
    
    def mostrar_visualizaciones_avanzadas(self, dataset: pd.DataFrame):
        """Muestra visualizaciones avanzadas del dataset"""
        st.subheader(t("eda.visualizaciones_avanzadas"))
        
        if not MODULOS_DISPONIBLES:
            st.warning("⚠️ Visualizaciones avanzadas requieren módulos completos")
            return
        
        # Crear tabs para diferentes tipos de visualización
        tab1, tab2, tab3 = st.tabs([t("eda.histogramas"), t("eda.box_plots"), t("eda.scatter_plots")])
        
        with tab1:
            st.markdown(f"### {t('eda.distribucion_caracteristicas')}")
            fig_hist = analizador_eda.crear_histogramas_caracteristicas(dataset)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            st.info("💡 Los histogramas muestran cómo se distribuyen las características de imagen en cada clase diagnóstica")
        
        with tab2:
            st.markdown(f"### {t('eda.analisis_distribucion')}")
            fig_box = analizador_eda.crear_boxplots_comparativos(dataset)
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.info("💡 Los box plots permiten identificar outliers y comparar la variabilidad entre clases")
        
        with tab3:
            st.markdown(f"### {t('eda.analisis_multivariable')}")
            fig_scatter = analizador_eda.crear_scatter_plot_multivariable(dataset)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("💡 El scatter plot muestra relaciones entre múltiples variables simultáneamente")
    
    def mostrar_insights_automaticos(self, dataset: pd.DataFrame):
        """Genera y muestra insights automáticos del dataset"""
        st.subheader(t("eda.insights_automaticos"))
        
        insights = []
        
        # Insight 1: Balance de clases
        distribucion = dataset['clase_diagnostico'].value_counts()
        balance_ratio = distribucion.min() / distribucion.max()
        
        if balance_ratio > 0.7:
            insights.append("✅ **Dataset bien balanceado**: Las clases tienen representación similar")
        elif balance_ratio > 0.3:
            insights.append("⚠️ **Dataset moderadamente desbalanceado**: Considerar técnicas de balanceo")
        else:
            insights.append("🚨 **Dataset muy desbalanceado**: Requiere estrategias especiales de entrenamiento")
        
        # Insight 2: Correlaciones significativas
        if len(dataset.select_dtypes(include=[np.number]).columns) > 1:
            variables_numericas = dataset.select_dtypes(include=[np.number]).columns
            correlaciones = dataset[variables_numericas].corr()
            
            # Encontrar correlaciones fuertes (excluyendo diagonal)
            mask = np.triu(np.ones_like(correlaciones, dtype=bool))
            correlaciones_altas = correlaciones.mask(mask).abs() > 0.7
            
            if correlaciones_altas.any().any():
                insights.append("🔗 **Correlaciones fuertes detectadas**: Algunas variables están altamente correlacionadas")
            else:
                insights.append("📊 **Variables independientes**: No se detectaron correlaciones muy fuertes")
        
        # Insight 3: Edad y diagnóstico
        if 'edad_paciente' in dataset.columns:
            edad_covid = dataset[dataset['clase_diagnostico'] == 'COVID-19']['edad_paciente'].mean()
            edad_normal = dataset[dataset['clase_diagnostico'] == 'Normal']['edad_paciente'].mean()
            
            if abs(edad_covid - edad_normal) > 10:
                insights.append(f"👥 **Diferencia etaria significativa**: COVID-19 promedio {edad_covid:.1f} años vs Normal {edad_normal:.1f} años")
        
        # Insight 4: Calidad de datos
        valores_faltantes_pct = (dataset.isnull().sum().sum() / dataset.size) * 100
        if valores_faltantes_pct == 0:
            insights.append("✨ **Datos perfectos**: Sin valores faltantes en el dataset")
        elif valores_faltantes_pct < 5:
            insights.append(f"✅ **Buena calidad**: Solo {valores_faltantes_pct:.1f}% de valores faltantes")
        else:
            insights.append(f"⚠️ **Atención requerida**: {valores_faltantes_pct:.1f}% de valores faltantes")
        
        # Mostrar insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"**{i}.** {insight}")
        
        # Recomendaciones
        st.markdown(f"### {t('eda.recomendaciones_modelado')}")
        
        recomendaciones = [
            "🎯 Considerar validación cruzada estratificada debido a múltiples clases",
            "📊 Evaluar feature engineering basado en las correlaciones encontradas",
            "🔄 Implementar data augmentation si el dataset es limitado",
            "📈 Usar métricas balanceadas (F1-score, AUC) para evaluación"
        ]
        
        for recomendacion in recomendaciones:
            st.markdown(f"- {recomendacion}")
    
    def generar_reporte_eda_pdf(self, dataset: pd.DataFrame):
        """Genera y descarga reporte PDF para EDA"""
        if not REPORTES_DISPONIBLES:
            st.error("❌ Generador de reportes no disponible")
            return
        
        try:
            with st.spinner(t("reportes.generando_reporte")):
                # Crear visualizaciones para el reporte
                visualizaciones = {}
                
                if MODULOS_DISPONIBLES:
                    visualizaciones = {
                        'mapa_calor': analizador_eda.crear_mapa_calor_correlaciones(dataset),
                        'histogramas': analizador_eda.crear_histogramas_caracteristicas(dataset),
                        'boxplots': analizador_eda.crear_boxplots_comparativos(dataset),
                        'scatter': analizador_eda.crear_scatter_plot_multivariable(dataset)
                    }
                
                # Generar PDF
                pdf_content = generador_reportes.generar_reporte_eda(dataset, visualizaciones)
                
                # Crear nombre de archivo localizado
                nombre_archivo = generador_reportes.crear_nombre_archivo('eda', 'pdf')
                
                # Botón de descarga
                st.download_button(
                    label=t("reportes.descargar_pdf"),
                    data=pdf_content,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    help=f"Descargar reporte EDA en formato PDF ({nombre_archivo})"
                )
                
                st.success(t("reportes.reporte_generado"))
                
        except Exception as e:
            st.error(f"{t('reportes.error_generacion')}: {str(e)}")
    
    def _traducir_variable(self, variable: str) -> str:
        """Traduce nombres de variables a español legible"""
        traduccion = {
            'opacidad_media': 'Opacidad Media',
            'contraste_medio': 'Contraste Medio',
            'textura_rugosidad': 'Rugosidad de Textura',
            'densidad_pulmonar': 'Densidad Pulmonar',
            'edad_paciente': 'Edad del Paciente',
            'entropia_imagen': 'Entropía',
            'energia_imagen': 'Energía',
            'homogeneidad': 'Homogeneidad'
        }
        return traduccion.get(variable, variable.replace('_', ' ').title())
    
    def _interpretar_correlacion(self, valor: float) -> str:
        """Interpreta el valor de correlación"""
        abs_valor = abs(valor)
        if abs_valor >= 0.8:
            return "Muy fuerte"
        elif abs_valor >= 0.6:
            return "Fuerte"
        elif abs_valor >= 0.4:
            return "Moderada"
        elif abs_valor >= 0.2:
            return "Débil"
        else:
            return "Muy débil"
    
    def ejecutar_dashboard_completo(self):
        """Ejecuta el dashboard EDA completo"""
        st.title(t("eda.titulo"))
        st.markdown("---")
        
        # Configuración del dataset
        with st.sidebar:
            st.header(t("eda.configuracion"))
            tamaño_muestra = st.slider(
                t("eda.tamaño_dataset"),
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Número de radiografías para el análisis"
            )
            
            regenerar = st.button(t("eda.regenerar_dataset"), help="Generar nuevo dataset para análisis")
        
        # Cargar dataset
        if regenerar or 'dataset_eda' not in st.session_state:
            with st.spinner("🔄 Generando dataset para análisis..."):
                st.session_state.dataset_eda = self.cargar_dataset_covid(tamaño_muestra)
            st.success(t("eda.dataset_generado"))
        
        dataset = st.session_state.dataset_eda
        
        # Secciones del dashboard
        self.mostrar_resumen_ejecutivo(dataset)
        st.markdown("---")
        
        self.mostrar_distribucion_clases(dataset)
        st.markdown("---")
        
        self.mostrar_estadisticos_descriptivos(dataset)
        st.markdown("---")
        
        self.mostrar_mapa_calor_correlaciones(dataset)
        st.markdown("---")
        
        self.mostrar_visualizaciones_avanzadas(dataset)
        st.markdown("---")
        
        self.mostrar_insights_automaticos(dataset)
        
        # Sección de reporte PDF
        st.markdown("---")
        st.subheader("📄 Reporte PDF del Análisis EDA")
        
        col_pdf1, col_pdf2 = st.columns(2)
        
        with col_pdf1:
            if st.button("📄 Generar Reporte EDA PDF", type="secondary", use_container_width=True):
                self.generar_reporte_eda_pdf(dataset)
        
        with col_pdf2:
            st.info("💡 El reporte incluye estadísticas completas, visualizaciones y insights automáticos del análisis exploratorio")
        
        # Footer
        st.markdown("---")
        st.info(t("eda.nota_requisitos"))

# Instancia global del dashboard
dashboard_eda = DashboardEDA()