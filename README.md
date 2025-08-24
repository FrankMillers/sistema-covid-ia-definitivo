# 🦠 Sistema IA COVID-19 - Detección Automatizada

## 📋 Descripción
Sistema de inteligencia artificial multilenguaje para detección automatizada de COVID-19 en radiografías de tórax usando 6 modelos específicos de Machine Learning y Deep Learning con análisis estadístico robusto incluyendo Matthews Correlation Coefficient (MCC) y Tests de McNemar.

## 🎯 Características Principales

### 🤖 **Modelos de Inteligencia Artificial**
- ✅ **6 Modelos Especializados**: Custom CNN, MobileNetV2, EfficientNet, Ensemble, CNN+XGBoost, CNN+RandomForest
- ✅ **Transfer Learning**: Arquitecturas pre-entrenadas optimizadas para COVID-19
- ✅ **Ensemble Learning**: Combinación inteligente de modelos para máxima precisión
- ✅ **Evaluación Robusta**: Métricas avanzadas con validación cruzada

### 📊 **Análisis Estadístico Avanzado**
- ✅ **Matthews Correlation Coefficient (MCC)**: Métrica balanceada para clasificación multiclase
- ✅ **Tests de McNemar**: Comparación estadística entre pares de modelos
- ✅ **Curvas ROC Multiclase**: Análisis de rendimiento por clase diagnóstica
- ✅ **Matrices de Confusión**: Visualización interactiva de errores y aciertos

### 🌍 **Sistema Multilenguaje**
- ✅ **Español** 🇪🇸 (Idioma principal)
- ✅ **English** 🇬🇧 (Traducción completa)
- ✅ **Français** 🇫🇷 (Traducción completa)
- ✅ **Interfaz Adaptativa**: Cambio dinámico de idioma en tiempo real

### 📱 **Aplicación Web Interactiva**
- ✅ **Streamlit Dashboard**: Interfaz profesional y responsiva
- ✅ **Análisis Individual**: Upload de radiografías para diagnóstico inmediato
- ✅ **Dashboard EDA**: Análisis exploratorio completo con visualizaciones
- ✅ **Comparación de Modelos**: Ranking automático y análisis estadístico

### 📄 **Reportes Profesionales**
- ✅ **Generación Automática PDF**: Reportes médicos con diseño profesional
- ✅ **Recomendaciones Clínicas**: Sugerencias específicas por diagnóstico
- ✅ **Análisis Estadístico**: Incluye MCC, McNemar y métricas detalladas
- ✅ **Multilenguaje**: Reportes en 3 idiomas

## 🔬 **Modelos Disponibles**

| Modelo | Tipo | Accuracy | MCC | Especialidad |
|--------|------|----------|-----|--------------|
| **Custom_CNN** | 🏆 Campeón | 87.44% | 0.828 | Red neuronal personalizada optimizada |
| **MobileNetV2** | 🥈 Segundo | 83.96% | 0.781 | Arquitectura eficiente para móviles |
| **Ensemble** | 🎯 Combinado | 86.23% | 0.815 | Combinación inteligente de modelos |
| **EfficientNet** | 📊 Escalable | 82.15% | 0.762 | Escalamiento compuesto optimizado |
| **CNN_XGBoost** | 🔗 Híbrido | 79.84% | 0.728 | Deep Learning + Gradient Boosting |
| **CNN_RandomForest** | 🌳 Ensemble | 78.92% | 0.715 | CNN con Random Forest |

## 🏥 **Clases Diagnósticas**

| Clase | Descripción | Características Radiológicas |
|-------|-------------|-------------------------------|
| 🦠 **COVID-19** | Infección por SARS-CoV-2 | Opacidades en vidrio esmerilado bilaterales |
| 🫁 **Opacidad Pulmonar** | Alteraciones pulmonares inespecíficas | Consolidaciones localizadas |
| ✅ **Normal** | Radiografía sin hallazgos patológicos | Parénquima pulmonar conservado |
| 🦠 **Neumonía Viral** | Infección viral no-COVID | Consolidaciones multilobares |

## 🚀 Inicio Rápido

### **Prerrequisitos**
- 🐍 Python 3.8 o superior
- 💾 8GB RAM mínimo (16GB recomendado)

### **1. Clonar Repositorio**
```bash
git clone https://github.com/tuusuario/sistema-covid-ia-definitivo.git
cd sistema-covid-ia-definitivo
```

### **2. Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### **3. Ejecutar Aplicación**
```bash
streamlit run app.py
```
🎉 **¡La aplicación estará disponible en https://sistema-covid-ia-definitivo-iynxxsgn6b47hhacfappgoh.streamlit.app**
🎉 **¡La aplicación estará disponible en http://localhost:8501!**

## 📖 **Instalación Completa**

### **Crear Entorno Virtual (Recomendado)**
```bash
# Con conda
conda create -n covid-ia python=3.9
conda activate covid-ia

# O con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### **Instalación de Dependencias Específicas**
```bash
# Core ML/DL
pip install tensorflow==2.13.0
pip install scikit-learn==1.3.0
pip install numpy pandas

# Visualización
pip install streamlit==1.25.0
pip install plotly==5.15.0
pip install seaborn matplotlib

# Procesamiento de imágenes
pip install opencv-python==4.8.0
pip install Pillow==10.0.0

# Reportes PDF
pip install reportlab==4.0.4

# Análisis estadístico
pip install scipy==1.11.1
```

## 📂 **Estructura del Proyecto**

```
sistema-covid-ia-definitivo/
├── 📁 datos/                    # Datasets COVID-19
│   ├── raw/                     # Datos originales
│   ├── processed/               # Datos procesados
│   └── synthetic/               # Datos sintéticos para demo
├── 📁 modelos/                  # 6 modelos entrenados
│   ├── entrenados/             # SavedModels de TensorFlow
│   ├── custom_cnn_savedmodel/
│   ├── mobilenetv2_savedmodel/
│   ├── ensemble_savedmodel/
│   ├── efficientnet_savedmodel/
│   ├── cnn_xgboost_savedmodel/
│   └── cnn_randomforest_savedmodel/
├── 📁 src/                      # Código fuente españolizado
│   ├── sistema_multilenguaje/   # Gestión de idiomas
│   ├── evaluacion_robusta/      # MCC y McNemar
│   ├── analisis_eda/           # Análisis exploratorio
│   ├── aplicacion_web/         # Dashboards Streamlit
│   └── reportes_pdf/           # Generación de reportes
├── 📁 traducciones/            # Soporte multilenguaje
│   ├── es.json                 # Español
│   ├── en.json                 # English
│   └── fr.json                 # Français
├── 📁 reportes/                # PDFs generados
├── 📁 documentacion/           # Documentación técnica
├── app.py                      # Aplicación principal Streamlit
├── requirements.txt            # Dependencias
└── README.md                   # Este archivo
```

## 🎮 **Guía de Uso**

### **1. Análisis Individual de Radiografías**
1. 🧭 Navegar → Sidebar → "🔬 Análisis Individual"
2. 🤖 Seleccionar modelo IA (recomendado: Custom_CNN)
3. 📤 Subir radiografía (formato PNG/JPG)
4. 🔍 Presionar "Analizar Radiografía"
5. 📊 Revisar resultados y probabilidades
6. 📄 Descargar reporte PDF con recomendaciones médicas

### **2. Dashboard de Análisis Exploratorio (EDA)**
1. 🧭 Navegar → Sidebar → "📊 Dashboard EDA"
2. ⚙️ Configurar tamaño del dataset
3. 📈 Explorar distribuciones y correlaciones
4. 🔥 Analizar mapas de calor
5. 🤖 Revisar insights automáticos
6. 📄 Generar reporte EDA en PDF

### **3. Comparación Robusta de Modelos**
1. 🧭 Navegar → Sidebar → "🏆 Comparación Modelos"
2. ✅ Seleccionar modelos a comparar
3. 🚀 Presionar "Ejecutar Comparación Completa"
4. 📊 Revisar ranking global con MCC
5. 🔬 Analizar Tests de McNemar
6. 📈 Explorar curvas ROC multiclase
7. 💡 Leer recomendaciones finales
8. 📄 Generar reporte comparativo

## 📊 **Análisis Estadístico Avanzado**

### **🔢 Matthews Correlation Coefficient (MCC)**
- **Propósito**: Métrica balanceada para evaluación de clasificación multiclase
- **Rango**: [-1, +1] donde +1 indica predicción perfecta
- **Ventaja**: Considera todos los elementos de la matriz de confusión
- **Implementación**: Integrado con 20% de peso en el score global

### **🔬 Tests de McNemar**
- **Propósito**: Comparación estadística rigurosa entre pares de modelos
- **Metodología**: Test chi-cuadrado con corrección de continuidad
- **Interpretación**: p < 0.05 indica diferencia estadísticamente significativa
- **Aplicación**: Todos los pares de modelos son comparados automáticamente

### **📈 Métricas Complementarias**
- **Accuracy**: Porcentaje de predicciones correctas
- **F1-Score**: Media armónica entre precisión y recall
- **AUC-ROC**: Área bajo la curva para cada clase diagnóstica
- **Precision/Recall**: Métricas específicas por clase

## 🛠️ **Stack Tecnológico**

### **🧠 Machine Learning & Deep Learning**
- **TensorFlow 2.x**: Framework principal para redes neuronales
- **scikit-learn**: Algoritmos ML clásicos y métricas
- **NumPy**: Computación numérica eficiente
- **OpenCV**: Procesamiento de imágenes médicas

### **📊 Visualización & Análisis**
- **Streamlit**: Framework web para aplicaciones de ML
- **Plotly**: Gráficos interactivos profesionales
- **Seaborn/Matplotlib**: Visualizaciones estadísticas
- **Pandas**: Manipulación y análisis de datos

### **📄 Generación de Reportes**
- **ReportLab**: Creación de PDFs profesionales
- **Sistema Multilenguaje**: Gestión de traducciones dinámicas

## 🎯 **Estado del Proyecto**

### **✅ Completado**
- ✅ Arquitectura base del sistema
- ✅ 6 modelos de IA implementados y funcionando
- ✅ Sistema multilenguaje (ES/EN/FR) completo
- ✅ Dashboard web interactivo con Streamlit
- ✅ Análisis estadístico con MCC y McNemar
- ✅ Generación automática de reportes PDF
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Comparación robusta de modelos

### **🔄 En Desarrollo**
- 🔄 Optimización de modelos con datos reales
- 🔄 Integración con APIs médicas
- 🔄 Deployment en cloud (Streamlit Cloud)
- 🔄 Validación clínica con especialistas

### **📅 Próximas Características**
- 📅 Modo offline para entornos hospitalarios
- 📅 API REST para integración con PACS
- 📅 Análisis de series temporales
- 📅 Explicabilidad con GRAD-CAM

## ⚠️ **Disclaimer Médico**

> **IMPORTANTE**: Este sistema está desarrollado exclusivamente con fines educativos y de investigación. NO debe ser utilizado como herramienta de diagnóstico médico primario. Siempre consulte con profesionales médicos calificados para diagnóstico y tratamiento. El sistema no reemplaza el criterio clínico profesional.

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 **Equipo de Desarrollo**

**Grupo 2 - Integrantes:**
- ALIPIO ESQUIVEL FRANK MILLER
- CASTAÑEDA COBEÑAS JORGE LUIS  
- VASQUEZ MORAN LIZARDO VIDAL

---

**🦠 Desarrollado para la detección médica de COVID-19 con Inteligencia Artificial 🤖**

*Sistema académico - No autorizado para uso clínico directo*
