"""
Dashboard de Comparación de Modelos COVID-19
Autor: Sistema IA COVID-19
Descripción: Comparación robusta y ranking automático de modelos con reportes PDF
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Agregar rutas para imports
sys.path.append('src')

try:
    from sistema_multilenguaje.sistema_multilenguaje import t, gestor_multilenguaje
    from evaluacion_robusta.evaluador_modelos_covid import evaluador_modelos
    MODULOS_DISPONIBLES = True
except ImportError as e:
    print(f"⚠️ Error importando módulos de comparación: {e}")
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

class ComparadorModelos:
    """Dashboard para comparación robusta de modelos COVID-19"""
    
    def __init__(self):
        """Inicializa el comparador de modelos"""
        self.modelos_disponibles = [
            "Custom_CNN", "MobileNetV2", "Ensemble", 
            "EfficientNet", "CNN_XGBoost", "CNN_RandomForest"
        ]
        self.comparacion_cache = None
        
    def mostrar_configuracion_comparacion(self):
        """Muestra controles de configuración para la comparación"""
        st.sidebar.header(t("comparacion.configuracion"))
        
        # Selección de modelos para comparar
        modelos_seleccionados = st.sidebar.multiselect(
            t("comparacion.modelos_comparar"),
            options=self.modelos_disponibles,
            default=self.modelos_disponibles,
            help="Selecciona los modelos que quieres incluir en la comparación"
        )
        
        # Tamaño del dataset de evaluación
        tamaño_evaluacion = st.sidebar.slider(
            t("comparacion.tamaño_evaluacion"),
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Número de muestras sintéticas para evaluación robusta"
        )
        
        # Botón para ejecutar comparación
        ejecutar_comparacion = st.sidebar.button(
            t("comparacion.ejecutar_comparacion"),
            type="primary",
            help="Inicia evaluación robusta de todos los modelos seleccionados"
        )
        
        return modelos_seleccionados, tamaño_evaluacion, ejecutar_comparacion
    
    def mostrar_ranking_global(self, comparacion_resultados):
        """Muestra ranking global de modelos"""
        st.subheader(t("comparacion.ranking_global"))
        
        ranking = comparacion_resultados['ranking_modelos']
        
        # Crear DataFrame para mostrar ranking
        df_ranking = pd.DataFrame(ranking)
        
        # Formatear columnas
        df_ranking['Score Global'] = df_ranking['score_global'].apply(lambda x: f"{x:.2f}%")
        df_ranking['Accuracy'] = df_ranking['accuracy'].apply(lambda x: f"{x:.3f}")
        df_ranking['F1-Score'] = df_ranking['f1_macro'].apply(lambda x: f"{x:.3f}")
        df_ranking['MCC'] = df_ranking['mcc'].apply(lambda x: f"{x:.3f}")
        df_ranking['AUC'] = df_ranking['auc_macro'].apply(lambda x: f"{x:.3f}")
        
        # Reordenar columnas para mostrar
        df_display = df_ranking[['posicion', 'modelo', 'Score Global', 'Accuracy', 'F1-Score', 'MCC', 'AUC']]
        df_display.columns = ['🥇 Pos.', 'Modelo', t("comparacion.score_global"), t("comparacion.precision"), t("comparacion.f1_score"), t("comparacion.mcc"), t("comparacion.auc")]
        
        # Mostrar tabla con estilo
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Destacar el mejor modelo
        mejor_modelo = ranking[0]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                t("comparacion.mejor_modelo"),
                mejor_modelo['modelo'],
                f"{t('comparacion.score')}: {mejor_modelo['score_global']:.2f}%"
            )
        
        with col2:
            st.metric(
                t("comparacion.accuracy_campeon"),
                f"{mejor_modelo['accuracy']:.3f}",
                f"{t('comparacion.mcc')}: {mejor_modelo['mcc']:.3f}"
            )
        
        with col3:
            diferencia_segundo = ranking[0]['score_global'] - ranking[1]['score_global'] if len(ranking) > 1 else 0
            st.metric(
                t("comparacion.ventaja_segundo"),
                f"+{diferencia_segundo:.2f}%",
                t("comparacion.puntos_diferencia")
            )
        
        # Gráfico de barras del ranking
        self._mostrar_grafico_ranking(ranking)
    
    def _mostrar_grafico_ranking(self, ranking):
        """Muestra gráfico de barras del ranking"""
        df_ranking = pd.DataFrame(ranking)
        
        # Crear gráfico de barras
        fig = px.bar(
            df_ranking,
            x='modelo',
            y='score_global',
            title=f'📊 {t("comparacion.score_global")} por Modelo',
            labels={'score_global': f'{t("comparacion.score_global")} (%)', 'modelo': 'Modelo'},
            color='score_global',
            color_continuous_scale='Viridis',
            text='score_global'
        )
        
        # Personalizar gráfico
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Modelos",
            yaxis_title=f"{t('comparacion.score_global')} (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def mostrar_metricas_detalladas(self, comparacion_resultados):
        """Muestra métricas detalladas de todos los modelos"""
        st.subheader(t("comparacion.metricas_detalladas"))
        
        evaluaciones = comparacion_resultados['evaluaciones_individuales']
        
        # Crear tabs para cada modelo
        nombres_modelos = list(evaluaciones.keys())
        tabs = st.tabs(nombres_modelos)
        
        for i, (modelo, tab) in enumerate(zip(nombres_modelos, tabs)):
            with tab:
                eval_modelo = evaluaciones[modelo]
                
                # Métricas principales en columnas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        t("comparacion.precision"),
                        f"{eval_modelo['metricas_basicas']['accuracy']:.3f}",
                        f"F1: {eval_modelo['metricas_basicas']['f1_macro']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        t("comparacion.mcc"),
                        f"{eval_modelo['mcc']:.3f}",
                        t("comparacion.matthews_correlation")
                    )
                
                with col3:
                    st.metric(
                        t("comparacion.auc_macro"),
                        f"{eval_modelo['auc_scores']['auc_macro']:.3f}",
                        f"{t('comparacion.weighted')}: {eval_modelo['auc_scores']['auc_weighted']:.3f}"
                    )
                
                with col4:
                    st.metric(
                        t("comparacion.score_global"),
                        f"{eval_modelo['score_global']:.2f}%",
                        t("comparacion.ponderado")
                    )
                
                # Métricas por clase
                st.markdown(f"### {t('comparacion.metricas_por_clase')}")
                
                clases = ['COVID-19', 'Opacidad Pulmonar', 'Normal', 'Neumonía Viral']
                metricas_clase = []
                
                for clase in clases:
                    clase_key = clase.lower().replace(' ', '_').replace('-', '_')
                    if f'precision_{clase_key}' in eval_modelo['metricas_basicas']:
                        metricas_clase.append({
                            'Clase': clase,
                            t('comparacion.precision'): f"{eval_modelo['metricas_basicas'][f'precision_{clase_key}']:.3f}",
                            t('comparacion.recall'): f"{eval_modelo['metricas_basicas'][f'recall_{clase_key}']:.3f}",
                            t('comparacion.f1_score'): f"{eval_modelo['metricas_basicas'][f'f1_{clase_key}']:.3f}",
                            t('comparacion.auc'): f"{eval_modelo['auc_scores'][f'auc_{clase_key}']:.3f}"
                        })
                
                if metricas_clase:
                    df_metricas = pd.DataFrame(metricas_clase)
                    st.dataframe(df_metricas, use_container_width=True, hide_index=True)
    
    def mostrar_matrices_confusion(self, comparacion_resultados):
        """Muestra matrices de confusión de todos los modelos"""
        st.subheader(t("comparacion.matrices_confusion"))
        
        evaluaciones = comparacion_resultados['evaluaciones_individuales']
        
        # Selector de modelo para matriz individual
        modelo_seleccionado = st.selectbox(
            t("comparacion.seleccionar_matriz"),
            options=list(evaluaciones.keys()),
            help="Elige un modelo para ver su matriz de confusión en detalle"
        )
        
        if modelo_seleccionado and modelo_seleccionado in evaluaciones:
            eval_modelo = evaluaciones[modelo_seleccionado]
            
            # Mostrar matriz de confusión individual
            st.plotly_chart(
                eval_modelo['matriz_confusion'], 
                use_container_width=True
            )
            
            # Interpretación automática
            st.markdown(f"### {t('comparacion.interpretacion_matriz')}")
            self._interpretar_matriz_confusion(modelo_seleccionado, evaluaciones)
    
    def _interpretar_matriz_confusion(self, modelo_nombre, evaluaciones):
        """Interpreta automáticamente la matriz de confusión"""
        eval_modelo = evaluaciones[modelo_nombre]
        metricas = eval_modelo['metricas_basicas']
        
        # Análisis automático
        clases = ['COVID-19', 'Opacidad Pulmonar', 'Normal', 'Neumonía Viral']
        interpretaciones = []
        
        for clase in clases:
            clase_key = clase.lower().replace(' ', '_').replace('-', '_')
            if f'precision_{clase_key}' in metricas and f'recall_{clase_key}' in metricas:
                precision = metricas[f'precision_{clase_key}']
                recall = metricas[f'recall_{clase_key}']
                
                if precision > 0.8 and recall > 0.8:
                    interpretaciones.append(f"✅ **{clase}**: {t('comparacion.excelente_rendimiento')} (P={precision:.3f}, R={recall:.3f})")
                elif precision > 0.6 and recall > 0.6:
                    interpretaciones.append(f"⚠️ **{clase}**: {t('comparacion.rendimiento_moderado')} (P={precision:.3f}, R={recall:.3f})")
                else:
                    interpretaciones.append(f"❌ **{clase}**: {t('comparacion.rendimiento_bajo')} (P={precision:.3f}, R={recall:.3f})")
        
        for interpretacion in interpretaciones:
            st.markdown(interpretacion)
        
        # Recomendaciones
        st.markdown(f"### {t('comparacion.recomendaciones')}")
        if metricas['accuracy'] > 0.85:
            st.success(t("comparacion.modelo_excelente"))
        elif metricas['accuracy'] > 0.75:
            st.warning(t("comparacion.modelo_aceptable"))
        else:
            st.error(t("comparacion.modelo_requiere"))
    
    def mostrar_curvas_roc(self, comparacion_resultados):
        """Muestra curvas ROC de todos los modelos"""
        st.subheader(t("comparacion.curvas_roc"))
        
        evaluaciones = comparacion_resultados['evaluaciones_individuales']
        
        # Selector de modelo para ROC individual
        modelo_seleccionado = st.selectbox(
            t("comparacion.seleccionar_roc"),
            options=list(evaluaciones.keys()),
            help="Elige un modelo para ver sus curvas ROC detalladas",
            key="roc_selector"
        )
        
        if modelo_seleccionado and modelo_seleccionado in evaluaciones:
            eval_modelo = evaluaciones[modelo_seleccionado]
            
            # Mostrar curvas ROC
            st.plotly_chart(
                eval_modelo['curvas_roc'], 
                use_container_width=True
            )
            
            # Tabla de AUC por clase
            st.markdown(f"### {t('comparacion.valores_auc')}")
            
            auc_scores = eval_modelo['auc_scores']
            clases = ['COVID-19', 'Opacidad Pulmonar', 'Normal', 'Neumonía Viral']
            
            auc_data = []
            for clase in clases:
                clase_key = clase.lower().replace(' ', '_').replace('-', '_')
                if f'auc_{clase_key}' in auc_scores:
                    auc_value = auc_scores[f'auc_{clase_key}']
                    interpretacion = self._interpretar_auc(auc_value)
                    auc_data.append({
                        'Clase': clase,
                        t('comparacion.auc'): f"{auc_value:.3f}",
                        t('comparacion.interpretacion'): interpretacion
                    })
            
            if auc_data:
                df_auc = pd.DataFrame(auc_data)
                st.dataframe(df_auc, use_container_width=True, hide_index=True)
    
    def _interpretar_auc(self, auc_value):
        """Interpreta el valor AUC"""
        if auc_value >= 0.9:
            return t("comparacion.excelente")
        elif auc_value >= 0.8:
            return t("comparacion.bueno")
        elif auc_value >= 0.7:
            return t("comparacion.aceptable")
        elif auc_value >= 0.6:
            return t("comparacion.pobre")
        else:
            return t("comparacion.muy_pobre")
    
    def mostrar_tests_mcnemar(self, comparacion_resultados):
        """Muestra resultados de tests de McNemar"""
        st.subheader(t("comparacion.tests_mcnemar"))
        
        tests_mcnemar = comparacion_resultados['tests_mcnemar']
        
        if not tests_mcnemar:
            st.warning("⚠️ No hay tests de McNemar disponibles")
            return
        
        st.markdown(t("comparacion.mcnemar_descripcion"))
        
        # Crear tabla de resultados
        resultados_mcnemar = []
        
        for comparacion, resultado in tests_mcnemar.items():
            modelo1, modelo2 = comparacion.split('_vs_')
            
            resultados_mcnemar.append({
                t('comparacion.comparacion'): f"{modelo1} vs {modelo2}",
                t('comparacion.estadistico'): f"{resultado['statistic']:.3f}",
                t('comparacion.p_valor'): f"{resultado['pvalue']:.6f}",
                t('comparacion.significativo'): t("comparacion.si") if resultado['significativo'] else t("comparacion.no"),
                t('comparacion.interpretacion'): resultado['interpretacion']
            })
        
        df_mcnemar = pd.DataFrame(resultados_mcnemar)
        st.dataframe(df_mcnemar, use_container_width=True, hide_index=True)
        
        # Resumen de significancia
        significativos = sum(1 for r in resultados_mcnemar if t("comparacion.si") in r[t('comparacion.significativo')])
        total = len(resultados_mcnemar)
        
        st.info(f"""
        {t("comparacion.resumen")}: {significativos}/{total} {t("comparacion.comparaciones_significativas")}
        
        {t("comparacion.interpretacion_mcnemar")}
        - {t("comparacion.significativo_descripcion")}
        - {t("comparacion.no_significativo_descripcion")}
        """)
    
    def mostrar_recomendaciones_finales(self, comparacion_resultados):
        """Muestra recomendaciones finales basadas en la comparación"""
        st.subheader(t("comparacion.recomendaciones_finales"))
        
        ranking = comparacion_resultados['ranking_modelos']
        mejor_modelo = ranking[0]
        
        # Análisis del mejor modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {t('comparacion.modelo_recomendado')}")
            st.success(f"""
            **{mejor_modelo['modelo']}** {t("comparacion.es_mejor")}
            
            🎯 **{t("comparacion.score_global")}**: {mejor_modelo['score_global']:.2f}%
            📊 **Accuracy**: {mejor_modelo['accuracy']:.3f}
            🔗 **MCC**: {mejor_modelo['mcc']:.3f}
            📈 **AUC Macro**: {mejor_modelo['auc_macro']:.3f}
            """)
        
        with col2:
            st.markdown(f"### {t('comparacion.justificacion_tecnica')}")
            self._generar_justificacion_tecnica(mejor_modelo, ranking)
        
        # Recomendaciones de uso
        st.markdown(f"### {t('comparacion.recomendaciones_uso')}")
        
        recomendaciones_uso = [
            f"{t('comparacion.produccion')} {mejor_modelo['modelo']} {t('comparacion.para_casos_criticos')}",
            f"{t('comparacion.backup')} {ranking[1]['modelo']} {t('comparacion.como_secundario')}",
            t("comparacion.ensemble"),
            t("comparacion.monitoreo"),
            t("comparacion.validacion")
        ]
        
        for recomendacion in recomendaciones_uso:
            st.markdown(f"- {recomendacion}")
        
        # Limitaciones y consideraciones
        st.markdown(f"### {t('comparacion.limitaciones')}")
        
        limitaciones = [
            t("comparacion.evaluacion_sintetica"),
            t("comparacion.contexto_clinico"),
            t("comparacion.actualizacion"),
            t("comparacion.metricas_especificas"),
            t("comparacion.uso_especifico")
        ]
        
        for limitacion in limitaciones:
            st.markdown(f"- {limitacion}")
    
    def _generar_justificacion_tecnica(self, mejor_modelo, ranking):
        """Genera justificación técnica del mejor modelo usando traducciones"""
        fortalezas = []
        
        if mejor_modelo['accuracy'] > 0.85:
            fortalezas.append(t("comparacion.alta_accuracy"))
        
        if mejor_modelo['mcc'] > 0.7:
            fortalezas.append(t("comparacion.excelente_balance"))
        
        if mejor_modelo['auc_macro'] > 0.8:
            fortalezas.append(t("comparacion.buena_discriminacion"))
        
        if len(ranking) > 1:
            diferencia = mejor_modelo['score_global'] - ranking[1]['score_global']
            if diferencia > 2:
                fortalezas.append(t("comparacion.ventaja_clara").format(diferencia=diferencia))
        
        if not fortalezas:
            fortalezas.append(t("comparacion.rendimiento_moderado_but"))
        
        for fortaleza in fortalezas:
            st.markdown(f"- {fortaleza}")
    
    def generar_reporte_comparacion_pdf(self, comparacion_resultados):
        """Genera y descarga reporte PDF para comparación de modelos"""
        if not REPORTES_DISPONIBLES:
            st.error("❌ Generador de reportes no disponible")
            return
        
        try:
            with st.spinner(t("reportes.generando_reporte")):
                # Generar PDF
                pdf_content = generador_reportes.generar_reporte_comparacion_modelos(comparacion_resultados)
                
                # Crear nombre de archivo localizado
                nombre_archivo = generador_reportes.crear_nombre_archivo('comparacion', 'pdf')
                
                # Botón de descarga
                st.download_button(
                    label=t("reportes.descargar_pdf"),
                    data=pdf_content,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    help=f"Descargar reporte de comparación en formato PDF ({nombre_archivo})"
                )
                
                st.success(t("reportes.reporte_generado"))
                
        except Exception as e:
            st.error(f"{t('reportes.error_generacion')}: {str(e)}")
    
    def ejecutar_dashboard_completo(self):
        """Ejecuta el dashboard completo de comparación de modelos"""
        st.title(t("comparacion.titulo"))
        st.markdown("---")
        
        if not MODULOS_DISPONIBLES:
            st.error("❌ Módulos de evaluación no disponibles")
            st.info("💡 Instale dependencias: scikit-learn, scipy")
            return
        
        # Configuración
        modelos_seleccionados, tamaño_evaluacion, ejecutar_comparacion = self.mostrar_configuracion_comparacion()
        
        if not modelos_seleccionados:
            st.warning("⚠️ Selecciona al menos un modelo para comparar")
            return
        
        # Ejecutar comparación
        if ejecutar_comparacion or 'comparacion_resultados' not in st.session_state:
            with st.spinner("🔄 Ejecutando evaluación robusta de modelos..."):
                try:
                    comparacion_resultados = evaluador_modelos.comparar_todos_los_modelos(
                        modelos_seleccionados, 
                        tamaño_evaluacion
                    )
                    st.session_state.comparacion_resultados = comparacion_resultados
                    st.success(t("comparacion.comparacion_completada"))
                except Exception as e:
                    st.error(f"❌ Error en comparación: {str(e)}")
                    return
        
        # Usar resultados cached o recién calculados
        comparacion_resultados = st.session_state.get('comparacion_resultados')
        
        if not comparacion_resultados:
            st.warning("⚠️ No hay resultados de comparación disponibles")
            return
        
        # Mostrar resultados
        self.mostrar_ranking_global(comparacion_resultados)
        st.markdown("---")
        
        self.mostrar_metricas_detalladas(comparacion_resultados)
        st.markdown("---")
        
        self.mostrar_matrices_confusion(comparacion_resultados)
        st.markdown("---")
        
        self.mostrar_curvas_roc(comparacion_resultados)
        st.markdown("---")
        
        self.mostrar_tests_mcnemar(comparacion_resultados)
        st.markdown("---")
        
        self.mostrar_recomendaciones_finales(comparacion_resultados)
        
        # Sección de reporte PDF
        st.markdown("---")
        st.subheader("📄 Reporte PDF de Comparación de Modelos")
        
        col_pdf1, col_pdf2 = st.columns(2)
        
        with col_pdf1:
            if st.button("📄 Generar Reporte Comparación PDF", type="secondary", use_container_width=True):
                self.generar_reporte_comparacion_pdf(comparacion_resultados)
        
        with col_pdf2:
            st.info("💡 El reporte incluye ranking completo, análisis estadístico, tests de McNemar y recomendaciones finales")
        
        # Footer
        st.markdown("---")
        st.info(t("comparacion.nota_requisitos"))

# Instancia global del comparador
comparador_modelos = ComparadorModelos()