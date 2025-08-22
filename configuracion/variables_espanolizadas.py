"""
Variables y constantes españolizadas para todo el proyecto
"""

# CONFIGURACIÓN MODELOS
NOMBRES_MODELOS = [
    'MobileNetV2_FineTuned',
    'Custom_CNN', 
    'EfficientNetB0_FineTuned',
    'CNN_XGBoost_Hybrid',
    'CNN_RandomForest_Hybrid',
    'Ensemble_Voting'
]

# CLASES COVID
CLASES_COVID = ['Normal', 'COVID-19', 'Neumonia']

# RUTAS DATOS
RUTA_DATOS_RAW = 'datos/raw/'
RUTA_DATOS_PROCESSED = 'datos/processed/'
RUTA_MODELOS = 'modelos/entrenados/'

# CONFIGURACIÓN ENTRENAMIENTO
TAMAÑO_IMAGEN = (224, 224)
BATCH_SIZE = 32
EPOCAS_ENTRENAMIENTO = 50
