"""
Dashboard de Comparación de Modelos COVID-19
Autor: Sistema IA COVID-19
Descripción: Comparación robusta y ranking automático de modelos
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
        st.sidebar.header("⚙️ Configuración Comparación")
        
        # Selección de modelos para comparar
        modelos_seleccionados = st.sidebar.multiselect(
            "Modelos a Comparar:",
            options=self.modelos_disponibles,
            default=self.modelos_disponibles,
            help="Selecciona los modelos que quieres incluir en la comparación"
        )
        
        # Tamaño del dataset de evaluación
        tamaño_evaluacion = st.sidebar.slider(
            "Tamaño Dataset Evaluación:",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Número de muestras sintéticas para evaluación robusta"
        )
        
        # Botón para ejecutar comparación
        ejecutar_comparacion = st.sidebar.button(
            "🚀 Ejecutar Comparación Completa",
            type="primary",
            help="Inicia evaluación robusta de todos los modelos seleccionados"
        )
        
        return modelos_seleccionados, tamaño_evaluacion, ejecutar_comparacion
    
    def mostrar_ranking_global(self, comparacion_resultados):
        """Muestra ranking global de modelos"""
        st.subheader("🏆 Ranking Global de Modelos")
        
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
        df_display.columns = ['🥇 Pos.', 'Modelo', 'Score Global', 'Accuracy', 'F1-Score', 'MCC', 'AUC']
        
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
                "🏆 Mejor Modelo",
                mejor_modelo['modelo'],
                f"Score: {mejor_modelo['score_global']:.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 Accuracy Campeón",
                f"{mejor_modelo['accuracy']:.3f}",
                f"MCC: {mejor_modelo['mcc']:.3f}"
            )
        
        with col3:
            diferencia_segundo = ranking[0]['score_global'] - ranking[1]['score_global'] if len(ranking) > 1 else 0
            st.metric(
                "📈 Ventaja sobre 2do",
                f"+{diferencia_segundo:.2f}%",
                "puntos de diferencia"
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
            title='📊 Score Global por Modelo',
            labels={'score_global': 'Score Global (%)', 'modelo': 'Modelo'},
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
            yaxis_title="Score Global (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def mostrar_metricas_detalladas(self, comparacion_resultados):
        """Muestra métricas detalladas de todos los modelos"""
        st.subheader("📊 Métricas Detalladas por Modelo")
        
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
                        "Accuracy",
                        f"{eval_modelo['metricas_basicas']['accuracy']:.3f}",
                        f"F1: {eval_modelo['metricas_basicas']['f1_macro']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "MCC",
                        f"{eval_modelo['mcc']:.3f}",
                        "Matthews Correlation"
                    )
                
                with col3:
                    st.metric(
                        "AUC Macro",
                        f"{eval_modelo['auc_scores']['auc_macro']:.3f}",
                        f"Weighted: {eval_modelo['auc_scores']['auc_weighted']:.3f}"
                    )
                
                with col4:
                    st.metric(
                        "Score Global",
                        f"{eval_modelo['score_global']:.2f}%",
                        "Ponderado"
                    )
                
                # Métricas por clase
                st.markdown("### 📋 Métricas por Clase")
                
                clases = ['COVID-19', 'Opacidad Pulmonar', 'Normal', 'Neumonía Viral']
                metricas_clase = []
                
                for clase in clases:
                    clase_key = clase.lower().replace(' ', '_').replace('-', '_')
                    if f'precision_{clase_key}' in eval_modelo['metricas_basicas']:
                        metricas_clase.append({
                            'Clase': clase,
                            'Precision': f"{eval_modelo['metricas_basicas'][f'precision_{clase_key}']:.3f}",
                            'Recall': f"{eval_modelo['metricas_basicas'][f'recall_{clase_key}']:.3f}",
                            'F1-Score': f"{eval_modelo['metricas_basicas'][f'f1_{clase_key}']:.3f}",
                            'AUC': f"{eval_modelo['auc_scores'][f'auc_{clase_key}']:.3f}"
                        })
                
                if metricas_clase:
                    df_metricas = pd.DataFrame(metricas_clase)
                    st.dataframe(df_metricas, use_container_width=True, hide_index=True)
    
    def mostrar_matrices_confusion(self, comparacion_resultados):
        """Muestra matrices de confusión de todos los modelos"""
        st.subheader("📊 Matrices de Confusión Comparativas")
        
        evaluaciones = comparacion_resultados['evaluaciones_individuales']
        
        # Selector de modelo para matriz individual
        modelo_seleccionado = st.selectbox(
            "Seleccionar Modelo para Matriz Detallada:",
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
            st.markdown("### 🔍 Interpretación de la Matriz")
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
                    interpretaciones.append(f"✅ **{clase}**: Excelente rendimiento (P={precision:.3f}, R={recall:.3f})")
                elif precision > 0.6 and recall > 0.6:
                    interpretaciones.append(f"⚠️ **{clase}**: Rendimiento moderado (P={precision:.3f}, R={recall:.3f})")
                else:
                    interpretaciones.append(f"❌ **{clase}**: Rendimiento bajo (P={precision:.3f}, R={recall:.3f})")
        
        for interpretacion in interpretaciones:
            st.markdown(interpretacion)
        
        # Recomendaciones
        st.markdown("### 💡 Recomendaciones")
        if metricas['accuracy'] > 0.85:
            st.success("🎯 Modelo con excelente rendimiento general")
        elif metricas['accuracy'] > 0.75:
            st.warning("📈 Modelo con rendimiento aceptable, considerar mejoras")
        else:
            st.error("🔧 Modelo requiere optimización significativa")
    
    def mostrar_curvas_roc(self, comparacion_resultados):
        """Muestra curvas ROC de todos los modelos"""
        st.subheader("📈 Curvas ROC Multiclase")
        
        evaluaciones = comparacion_resultados['evaluaciones_individuales']
        
        # Selector de modelo para ROC individual
        modelo_seleccionado = st.selectbox(
            "Seleccionar Modelo para Curvas ROC:",
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
            st.markdown("### 📊 Valores AUC por Clase")
            
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
                        'AUC': f"{auc_value:.3f}",
                        'Interpretación': interpretacion
                    })
            
            if auc_data:
                df_auc = pd.DataFrame(auc_data)
                st.dataframe(df_auc, use_container_width=True, hide_index=True)
    
    def _interpretar_auc(self, auc_value):
        """Interpreta el valor AUC"""
        if auc_value >= 0.9:
            return "🌟 Excelente"
        elif auc_value >= 0.8:
            return "✅ Bueno"
        elif auc_value >= 0.7:
            return "⚠️ Aceptable"
        elif auc_value >= 0.6:
            return "❌ Pobre"
        else:
            return "🚨 Muy pobre"
    
    def mostrar_tests_mcnemar(self, comparacion_resultados):
        """Muestra resultados de tests de McNemar"""
        st.subheader("🔬 Tests de McNemar - Comparación Estadística")
        
        tests_mcnemar = comparacion_resultados['tests_mcnemar']
        
        if not tests_mcnemar:
            st.warning("⚠️ No hay tests de McNemar disponibles")
            return
        
        st.markdown("""
        El **Test de McNemar** determina si hay diferencias **estadísticamente significativas** 
        entre el rendimiento de dos modelos en el mismo dataset.
        """)
        
        # Crear tabla de resultados
        resultados_mcnemar = []
        
        for comparacion, resultado in tests_mcnemar.items():
            modelo1, modelo2 = comparacion.split('_vs_')
            
            resultados_mcnemar.append({
                'Comparación': f"{modelo1} vs {modelo2}",
                'Estadístico': f"{resultado['statistic']:.3f}",
                'P-valor': f"{resultado['pvalue']:.6f}",
                'Significativo': "✅ Sí" if resultado['significativo'] else "❌ No",
                'Interpretación': resultado['interpretacion']
            })
        
        df_mcnemar = pd.DataFrame(resultados_mcnemar)
        st.dataframe(df_mcnemar, use_container_width=True, hide_index=True)
        
        # Resumen de significancia
        significativos = sum(1 for r in resultados_mcnemar if "✅" in r['Significativo'])
        total = len(resultados_mcnemar)
        
        st.info(f"""
        📊 **Resumen**: {significativos}/{total} comparaciones muestran diferencias estadísticamente significativas.
        
        💡 **Interpretación**: 
        - ✅ **Significativo** (p < 0.05): Los modelos tienen rendimiento estadísticamente diferente
        - ❌ **No significativo** (p ≥ 0.05): No hay evidencia de diferencia en rendimiento
        """)
    
    def mostrar_recomendaciones_finales(self, comparacion_resultados):
        """Muestra recomendaciones finales basadas en la comparación"""
        st.subheader("💡 Recomendaciones Finales y Conclusiones")
        
        ranking = comparacion_resultados['ranking_modelos']
        mejor_modelo = ranking[0]
        
        # Análisis del mejor modelo
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Modelo Recomendado")
            st.success(f"""
            **{mejor_modelo['modelo']}** es el modelo con mejor rendimiento global.
            
            🎯 **Score Global**: {mejor_modelo['score_global']:.2f}%
            📊 **Accuracy**: {mejor_modelo['accuracy']:.3f}
            🔗 **MCC**: {mejor_modelo['mcc']:.3f}
            📈 **AUC Macro**: {mejor_modelo['auc_macro']:.3f}
            """)
        
        with col2:
            st.markdown("### 📋 Justificación Técnica")
            self._generar_justificacion_tecnica(mejor_modelo, ranking)
        
        # Recomendaciones de uso
        st.markdown("### 🎯 Recomendaciones de Uso")
        
        recomendaciones_uso = [
            f"🥇 **Producción**: Usar {mejor_modelo['modelo']} para casos críticos de diagnóstico",
            f"🥈 **Backup**: Mantener {ranking[1]['modelo']} como modelo secundario",
            "🔄 **Ensemble**: Considerar combinación de los top 3 modelos para mayor robustez",
            "📊 **Monitoreo**: Implementar métricas de drift para detectar degradación del modelo",
            "🔍 **Validación**: Realizar validación adicional con datos reales antes de deployment"
        ]
        
        for recomendacion in recomendaciones_uso:
            st.markdown(f"- {recomendacion}")
        
        # Limitaciones y consideraciones
        st.markdown("### ⚠️ Limitaciones y Consideraciones")
        
        limitaciones = [
            "📊 **Evaluación sintética**: Resultados basados en datos simulados, validar con datos reales",
            "🏥 **Contexto clínico**: Considerar siempre el juicio médico profesional",
            "🔄 **Actualización**: Re-evaluar modelos periódicamente con nuevos datos",
            "📈 **Métricas específicas**: Priorizar recall para COVID-19 en contextos de screening",
            "🎯 **Uso específico**: Adaptar selección de modelo según caso de uso específico"
        ]
        
        for limitacion in limitaciones:
            st.markdown(f"- {limitacion}")
    
    def _generar_justificacion_tecnica(self, mejor_modelo, ranking):
        """Genera justificación técnica del mejor modelo"""
        fortalezas = []
        
        if mejor_modelo['accuracy'] > 0.85:
            fortalezas.append("✅ Alta accuracy general")
        
        if mejor_modelo['mcc'] > 0.7:
            fortalezas.append("✅ Excelente balance de clases (MCC)")
        
        if mejor_modelo['auc_macro'] > 0.8:
            fortalezas.append("✅ Buena discriminación (AUC)")
        
        if len(ranking) > 1:
            diferencia = mejor_modelo['score_global'] - ranking[1]['score_global']
            if diferencia > 2:
                fortalezas.append(f"✅ Ventaja clara (+{diferencia:.1f}%)")
        
        if not fortalezas:
            fortalezas.append("⚠️ Rendimiento moderado pero mejor disponible")
        
        for fortaleza in fortalezas:
            st.markdown(f"- {fortaleza}")
    
    def ejecutar_dashboard_completo(self):
        """Ejecuta el dashboard completo de comparación de modelos"""
        st.title("🏆 Comparación Robusta de Modelos COVID-19")
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
                    st.success("✅ Comparación completada exitosamente")
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
        
        # Footer
        st.markdown("---")
        st.info("📋 **Nota**: Esta comparación cumple con los requisitos 10, 11 y 12 del proyecto")

# Instancia global del comparador
comparador_modelos = ComparadorModelos()