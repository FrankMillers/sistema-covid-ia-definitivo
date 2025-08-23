import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import sys
import base64
from datetime import datetime

# Agregar src al path para imports
sys.path.append('src')

# Importar sistema multilenguaje
try:
    from sistema_multilenguaje.sistema_multilenguaje import (
        t, crear_selector_idioma, obtener_traducciones_modelos, 
        gestor_multilenguaje
    )
    MULTILENGUAJE_DISPONIBLE = True
except ImportError:
    print("⚠️ Sistema multilenguaje no disponible, usando textos por defecto")
    MULTILENGUAJE_DISPONIBLE = False
    
    # Funciones fallback
    def t(clave): return clave
    def crear_selector_idioma(): return 'es'
    def obtener_traducciones_modelos(): return {}

# Importar generador de reportes PDF
try:
    from reportes_pdf.generador_reportes import generador_reportes
    REPORTES_DISPONIBLES = True
except ImportError:
    print("⚠️ Generador de reportes PDF no disponible")
    REPORTES_DISPONIBLES = False

# Importar dashboard EDA
try:
    from aplicacion_web.dashboard_eda import dashboard_eda
    EDA_DISPONIBLE = True
except ImportError:
    print("⚠️ Dashboard EDA no disponible")
    EDA_DISPONIBLE = False

# Importar comparador de modelos
try:
    from aplicacion_web.comparador_modelos import comparador_modelos
    COMPARACION_DISPONIBLE = True
except ImportError:
    print("⚠️ Comparador de modelos no disponible")
    COMPARACION_DISPONIBLE = False

# Configuración
st.set_page_config(
    page_title="🦠 Sistema IA COVID-19",
    page_icon="🦠",
    layout="wide"
)

# Rutas modelos
RUTAS_MODELOS = {
    "Custom_CNN": "modelos/entrenados/custom_cnn_savedmodel/",
    "MobileNetV2": "modelos/entrenados/mobilenetv2_savedmodel/",
    "Ensemble": "modelos/entrenados/ensemble_savedmodel/",
    "EfficientNet": "modelos/entrenados/efficientnet_savedmodel/",
    "CNN_XGBoost": "modelos/entrenados/cnn_xgboost_savedmodel/",
    "CNN_RandomForest": "modelos/entrenados/cnn_randomforest_savedmodel/"
}

RENDIMIENTO_MODELOS = {
    "Custom_CNN": {"accuracy": 87.44, "mcc": 0.828, "tipo": "🏆 Campeón"},
    "MobileNetV2": {"accuracy": 83.96, "mcc": 0.781, "tipo": "🥈 Segundo"},
    "Ensemble": {"accuracy": 86.23, "mcc": 0.815, "tipo": "🎯 Ensemble"},
    "EfficientNet": {"accuracy": 82.15, "mcc": 0.762, "tipo": "📊 EfficientNet"},
    "CNN_XGBoost": {"accuracy": 79.84, "mcc": 0.728, "tipo": "🔗 Híbrido XGB"},
    "CNN_RandomForest": {"accuracy": 78.92, "mcc": 0.715, "tipo": "🌳 Híbrido RF"}
}

@st.cache_resource
def cargar_savedmodel(nombre_modelo):
    """Carga SavedModel correctamente"""
    try:
        ruta_modelo = RUTAS_MODELOS.get(nombre_modelo)
        if ruta_modelo and os.path.exists(ruta_modelo):
            # Cargar SavedModel genérico
            loaded_model = tf.saved_model.load(ruta_modelo)
            
            # Obtener signature para predicción
            if 'serving_default' in loaded_model.signatures:
                signature = loaded_model.signatures['serving_default']
                info = RENDIMIENTO_MODELOS[nombre_modelo]
                mensaje_exito = f"✅ {nombre_modelo} {info['tipo']} {t('modelo_cargado')} {info['accuracy']:.1f}%"
                return signature, mensaje_exito
            else:
                return None, f"❌ No hay signature serving_default en {nombre_modelo}"
        else:
            return None, f"❌ {t('modelo_no_encontrado')}: {ruta_modelo}"
    except Exception as e:
        return None, f"❌ {t('error_carga_modelo')} {nombre_modelo}: {str(e)}"

def predecir_covid(imagen, signature_func, nombre_modelo):
    """Predice COVID-19 usando SavedModel signature"""
    try:
        # Preprocesar imagen
        img_resized = cv2.resize(imagen, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        # Convertir a tensor
        input_tensor = tf.constant(img_array)
        
        # Hacer predicción usando signature
        resultado = signature_func(keras_tensor=input_tensor)
        
        # Extraer predicción del resultado
        prediccion = resultado['output_0'].numpy()
        
        # Clases COVID-19 traducidas
        clases = [t('covid19'), t('opacidad_pulmonar'), t('normal'), t('neumonia_viral')]
        
        clase_predicha = clases[np.argmax(prediccion)]
        confianza = np.max(prediccion) * 100
        
        return clase_predicha, confianza, prediccion[0]
        
    except Exception as e:
        st.error(f"❌ {t('error_procesamiento')} {nombre_modelo}: {str(e)}")
        return "Error", 0, None

def mostrar_recomendaciones_medicas(clase_predicha, confianza, idioma_actual):
    """Muestra recomendaciones médicas basadas en la predicción usando traducciones"""
    st.subheader(t("recomendaciones_medicas"))
    
    # Mapear clase predicha a clave de recomendación
    mapeo_clases = {
        'COVID-19': 'covid19',
        'Opacidad Pulmonar': 'opacidad_pulmonar', 
        'Lung Opacity': 'opacidad_pulmonar',
        'Opacité Pulmonaire': 'opacidad_pulmonar',
        'Normal': 'normal',
        'Neumonía Viral': 'neumonia_viral',
        'Viral Pneumonia': 'neumonia_viral',
        'Pneumonie Virale': 'neumonia_viral'
    }
    
    clave_recomendacion = mapeo_clases.get(clase_predicha, 'normal')
    
    # Obtener recomendaciones traducidas
    if MULTILENGUAJE_DISPONIBLE:
        recomendaciones_completas = gestor_multilenguaje.traducciones.get(idioma_actual, {}).get('recomendaciones', {})
    else:
        st.warning("⚠️ Sistema de recomendaciones no disponible")
        return
    
    # Determinar qué recomendaciones mostrar según confianza
    if clave_recomendacion == 'covid19':
        if confianza >= 70:
            clave_final = 'covid19_alta_confianza'
            st.error(f"🦠 **{t('detectado')}: {clase_predicha}** - {t('confianza')}: {confianza:.1f}%")
        else:
            clave_final = 'covid19_baja_confianza'
            st.warning(f"⚠️ **Posible {clase_predicha}** - {t('confianza')}: {confianza:.1f}%")
    else:
        clave_final = clave_recomendacion
        if clave_recomendacion == 'normal':
            st.success(f"✅ **{t('resultado')}: {clase_predicha}** - {t('confianza')}: {confianza:.1f}%")
        else:
            st.warning(f"⚠️ **{t('detectado')}: {clase_predicha}** - {t('confianza')}: {confianza:.1f}%")
    
    recomendaciones = recomendaciones_completas.get(clave_final, {})
    
    if not recomendaciones:
        st.warning("⚠️ No hay recomendaciones específicas disponibles para esta clasificación")
        return
    
    # Mostrar acciones inmediatas
    if 'acciones_inmediatas' in recomendaciones:
        st.markdown(f"### {t('acciones_inmediatas')}")
        for accion in recomendaciones['acciones_inmediatas']:
            st.markdown(f"- {accion}")
    
    # Mostrar exámenes adicionales
    if 'examenes_adicionales' in recomendaciones:
        st.markdown(f"### {t('examenes_adicionales')}")
        for examen in recomendaciones['examenes_adicionales']:
            st.markdown(f"- {examen}")
    
    # Mostrar seguimiento
    if 'seguimiento' in recomendaciones:
        st.markdown(f"### {t('seguimiento')}")
        for item in recomendaciones['seguimiento']:
            st.markdown(f"- {item}")
    
    # Mostrar recomendaciones generales
    if 'recomendaciones' in recomendaciones:
        for recomendacion in recomendaciones['recomendaciones']:
            st.info(recomendacion)

def generar_reporte_analisis_individual(resultado_analisis):
    """Genera y descarga reporte PDF para análisis individual - SIN REINICIOS"""
    if not REPORTES_DISPONIBLES:
        st.error("❌ Sistema de reportes PDF no disponible")
        st.code("pip install reportlab==4.0.4")
        return
    
    try:
        # Importar generador directamente
        from reportes_pdf.generador_reportes import generador_reportes
        
        # Generar PDF
        pdf_content = generador_reportes.generar_reporte_analisis_individual(resultado_analisis)
        
        # Crear nombre de archivo localizado
        nombre_archivo = generador_reportes.crear_nombre_archivo('analisis', 'pdf')
        
        # Crear key único para evitar reinicios
        import hashlib
        content_hash = hashlib.md5(str(resultado_analisis).encode()).hexdigest()[:8]
        
        # Botón de descarga directo
        st.download_button(
            label=t("reportes.descargar_pdf"),
            data=pdf_content,
            file_name=nombre_archivo,
            mime="application/pdf",
            key=f"pdf_analisis_{content_hash}",
            help=f"Descargar reporte en formato PDF ({nombre_archivo})"
        )
        
    except Exception as e:
        st.error(f"{t('reportes.error_generacion')}: {str(e)}")
        import traceback
        with st.expander("🔍 Ver detalles del error"):
            st.code(traceback.format_exc())

def crear_navegacion_principal():
    """Crea sistema de navegación principal de la aplicación con traducciones"""
    # Opciones de navegación usando traducciones
    opciones_nav = {
        'analisis_individual': t("navegacion.analisis_individual"),
        'dashboard_eda': t("navegacion.dashboard_eda"),
        'comparacion_modelos': t("navegacion.comparacion_modelos")
    }
    
    # Selector en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('navegacion.titulo')}")
    
    pagina_seleccionada = st.sidebar.radio(
        t("navegacion.seleccionar"),
        options=list(opciones_nav.keys()),
        format_func=lambda x: opciones_nav[x],
        key="navegacion_principal"
    )
    
    return pagina_seleccionada

def pagina_analisis_individual():
    """Página de análisis individual de radiografías"""
    # Header con texto traducido
    st.title(t("titulo_principal"))
    st.subheader(t("subtitulo"))
    
    # Sidebar - Configuración del modelo
    st.sidebar.header(t("configuracion_modelo"))
    
    # Obtener traducciones de modelos
    traducciones_modelos = obtener_traducciones_modelos()
    
    modelo_seleccionado = st.sidebar.selectbox(
        t("seleccionar_modelo"),
        list(RUTAS_MODELOS.keys()),
        help=t("ayuda_modelo")
    )
    
    # Mostrar info del modelo seleccionado con traducción
    info_modelo = RENDIMIENTO_MODELOS[modelo_seleccionado]
    tipo_modelo = traducciones_modelos.get(modelo_seleccionado, info_modelo['tipo'])
    
    st.sidebar.info(f"""
    **{modelo_seleccionado}**
    {tipo_modelo}
    
    {t("metricas")}
    - {t("accuracy")}: {info_modelo['accuracy']:.1f}%
    - MCC: {info_modelo['mcc']:.3f}
    """)
    
    # Cargar modelo seleccionado
    with st.spinner(f"{t('cargando_modelo')} {modelo_seleccionado}..."):
        signature, status = cargar_savedmodel(modelo_seleccionado)
    
    if signature is None:
        st.error(status)
        st.stop()
    
    st.success(status)
    
    # Sección principal - Upload de imagen
    st.header(t("analisis_radiografia"))
    
    uploaded_file = st.file_uploader(
        t("subir_radiografia"),
        type=['png', 'jpg', 'jpeg'],
        help=t("formatos_soportados")
    )
    
    if uploaded_file is not None:
        # Mostrar imagen subida
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(t("imagen_original"))
            st.image(image, caption=f"{t('archivo')}: {uploaded_file.name}", use_column_width=True)
            
            # Info de la imagen traducida
            st.info(f"""
            **{t('informacion_imagen')}**
            - {t('dimensiones')}: {image.size[0]} x {image.size[1]} px
            - {t('formato')}: {image.format}
            - {t('modo')}: {image.mode}
            """)
        
        with col2:
            st.subheader(t("analisis_modelo"))
            
            if st.button(t("analizar_radiografia"), type="primary", use_container_width=True):
                with st.spinner(t("analizando")):
                    # Convertir imagen a array
                    img_array = np.array(image.convert('RGB'))
                    
                    # Hacer predicción
                    clase, confianza, todas_predicciones = predecir_covid(
                        img_array, signature, modelo_seleccionado
                    )
                    
                    if clase != "Error":
                        # Mostrar resultado principal
                        if 'COVID' in clase.upper():
                            st.error(f"🦠 **{t('detectado')}: {clase}**")
                            st.error(f"{t('confianza')}: {confianza:.1f}%")
                        elif t('normal').upper() in clase.upper():
                            st.success(f"✅ **{t('resultado')}: {clase}**")
                            st.success(f"{t('confianza')}: {confianza:.1f}%")
                        else:
                            st.warning(f"⚠️ **{t('detectado')}: {clase}**")
                            st.warning(f"{t('confianza')}: {confianza:.1f}%")
                        
                        # Mostrar todas las probabilidades
                        st.subheader(t("probabilidades_detalladas"))
                        clases_nombres = [t('covid19'), t('opacidad_pulmonar'), t('normal'), t('neumonia_viral')]
                        
                        for i, (nombre, prob) in enumerate(zip(clases_nombres, todas_predicciones)):
                            porcentaje = prob * 100
                            if i == np.argmax(todas_predicciones):
                                st.metric(
                                    label=f"🎯 {nombre}",
                                    value=f"{porcentaje:.1f}%",
                                    delta=t("prediccion_principal")
                                )
                            else:
                                st.metric(
                                    label=nombre,
                                    value=f"{porcentaje:.1f}%"
                                )
                        
                        # Mostrar barra de progreso
                        st.subheader(t("distribucion_probabilidades"))
                        for nombre, prob in zip(clases_nombres, todas_predicciones):
                            st.progress(float(prob), text=f"{nombre}: {prob*100:.1f}%")
                        
                        # Guardar resultado para reporte PDF
                        resultado_analisis = {
                            'modelo_usado': modelo_seleccionado,
                            'clase_predicha': clase,
                            'confianza': confianza,
                            'probabilidades': todas_predicciones,
                            'archivo_original': uploaded_file.name,
                            'dimensiones_imagen': image.size,
                            'formato_imagen': image.format
                        }
                        
                        # Guardar en session state para reporte
                        st.session_state.ultimo_analisis = resultado_analisis
                        
                        # Mostrar recomendaciones médicas
                        st.markdown("---")
                        idioma_actual = gestor_multilenguaje.obtener_idioma_actual() if MULTILENGUAJE_DISPONIBLE else 'es'
                        mostrar_recomendaciones_medicas(clase, confianza, idioma_actual)
                        
                        # Sección de reporte PDF - SIN BOTÓN INTERMEDIO
                        st.markdown("---")
                        st.subheader("📄 Reporte PDF del Análisis")
                        
                        col_pdf1, col_pdf2 = st.columns(2)
                        
                        with col_pdf1:
                            # Mostrar botón de descarga directamente
                            generar_reporte_analisis_individual(resultado_analisis)
                        
                        with col_pdf2:
                            st.info("💡 El reporte incluye análisis completo, recomendaciones médicas y limitaciones del sistema")

def main():
    """Función principal de la aplicación"""
    # Selector de idioma en sidebar
    if MULTILENGUAJE_DISPONIBLE:
        idioma_actual = crear_selector_idioma()
    else:
        idioma_actual = 'es'
    
    # Sistema de navegación
    pagina = crear_navegacion_principal()
    
    # Ejecutar página seleccionada
    if pagina == 'analisis_individual':
        pagina_analisis_individual()
        
    elif pagina == 'dashboard_eda':
        if EDA_DISPONIBLE:
            dashboard_eda.ejecutar_dashboard_completo()
        else:
            st.error("❌ Dashboard EDA no disponible. Verifique las dependencias.")
            st.info("💡 Instale las dependencias faltantes: plotly, seaborn, pandas")
            
            # Mostrar detalles del error
            with st.expander("🔍 Detalles del Error"):
                st.code("""
                # Para instalar dependencias faltantes:
                pip install plotly seaborn pandas matplotlib
                
                # O actualizar requirements.txt:
                pip install -r requirements.txt
                """)
    
    elif pagina == 'comparacion_modelos':
        if COMPARACION_DISPONIBLE:
            comparador_modelos.ejecutar_dashboard_completo()
        else:
            st.error("❌ Comparador de modelos no disponible. Verifique las dependencias.")
            st.info("💡 Instale las dependencias faltantes: scikit-learn, scipy")
            
            # Mostrar detalles del error
            with st.expander("🔍 Detalles del Error"):
                st.code("""
                # Para instalar dependencias faltantes:
                pip install scikit-learn scipy
                
                # O actualizar requirements.txt:
                pip install -r requirements.txt
                """)
    
    else:
        st.error(f"❌ Página '{pagina}' no implementada")
    
    # Footer común a todas las páginas
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center'>
        <h4>{t("sistema_deteccion")}</h4>
        <p>{t("desarrollado_con")}</p>
        <p><em>{t("advertencia_medica")}</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()