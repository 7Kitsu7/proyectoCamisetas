import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image
import os
import base64
from fpdf import FPDF
from datetime import datetime
import tempfile
import json
from fpdf.enums import XPos, YPos
# Configuración de la página
st.set_page_config(page_title="Clasificador de Camisetas", page_icon="👕", layout="wide")

# ======================
# SISTEMA DE IDIOMAS
# ======================
# Diccionarios de traducción
TRANSLATIONS = {
    'es': {
        'app_title': "👕 Clasificador de Atributos de Camisetas",
        'app_description': "Suba una imagen de una camiseta para analizar sus atributos:",
        'gender': "Género",
        'usage': "Uso",
        'upload_label': "Seleccione una imagen de camiseta",
        'image_uploaded': "Imagen cargada",
        'analyze_button': "Analizar imagen",
        'analyzing': "Analizando imagen...",
        'analysis_complete': "¡Análisis completado con éxito!",
        'analyzed_image': "Imagen Analizada",
        'analysis_results': "Resultados del Análisis",
        'details': "Detalles de {attribute}",
        'generate_report': "Generar Reporte",
        'generating_pdf': "Generando documento PDF...",
        'report_success': "✅ Reporte generado con éxito!",
        'download_report': "⬇️ Descargar Reporte Completo",
        'pdf_report_name': "reporte_camiseta",
        'report_error': "❌ Error al generar el reporte: {error}",
        'about_title': "Acerca de esta aplicación",
        'about_content': """
Esta herramienta utiliza inteligencia artificial para analizar atributos de camisetas.

**Atributos que puede identificar:**
- **Género**: Hombre / Mujer
- **Uso**: Casual / Deportivo

Suba una imagen clara de una camiseta para obtener el análisis.
""",
        'men': "Hombre",
        'women': "Mujer",
        'casual': "Casual",
        'sports': "Deportivo",
        'processing_error': "❌ Error al procesar la imagen: {error}",
        'model_error': "Error al cargar el modelo: {error}"
    },
    'en': {
        'app_title': "👕 T-shirt Attribute Classifier",
        'app_description': "Upload a t-shirt image to analyze its attributes:",
        'gender': "Gender",
        'usage': "Usage",
        'upload_label': "Select a t-shirt image",
        'image_uploaded': "Image uploaded",
        'analyze_button': "Analyze image",
        'analyzing': "Analyzing image...",
        'analysis_complete': "Analysis completed successfully!",
        'analyzed_image': "Analyzed Image",
        'analysis_results': "Analysis Results",
        'details': "Details about {attribute}",
        'generate_report': "Generate Report",
        'generating_pdf': "Generating PDF document...",
        'report_success': "✅ Report generated successfully!",
        'download_report': "⬇️ Download Full Report",
        'pdf_report_name': "t-shirt_report",
        'report_error': "❌ Error generating report: {error}",
        'about_title': "About this app",
        'about_content': """
This tool uses artificial intelligence to analyze t-shirt attributes.

**Attributes it can identify:**
- **Gender**: Men / Women
- **Usage**: Casual / Sports

Upload a clear image of a t-shirt to get the analysis.
""",
        'men': "Men",
        'women': "Women",
        'casual': "Casual",
        'sports': "Sports",
        'processing_error': "❌ Error processing image: {error}",
        'model_error': "Error loading model: {error}"
    }
}

# Traducciones para el reporte PDF
PDF_TRANSLATIONS = {
    'es': {
        'report_title': "Reporte de Análisis de Camiseta",
        'analysis_date': "Fecha del análisis: {date}",
        'model_used': "Modelo utilizado: {model}",
        'analyzed_image': "Imagen analizada:",
        'main_results': "Resultados principales:",
        'attribute': "Atributo",
        'prediction': "Predicción",
        'confidence': "Confianza",
        'attribute_details': "Detalles por atributo:",
        'category': "Categoría",
        'probability': "Probabilidad",
        'model_evaluation': "Evaluación de Modelos",
        'evaluation_description': "Esta sección muestra las matrices de confusión y métricas de los diferentes modelos evaluados para la clasificación de camisetas.",
        'epochs': "Épocas:",
        'training': "Entrenamiento:",
        'validation': "Validación:",
        'testing': "Prueba:",
        'metrics_for': "Métricas para {attribute}:",
        'accuracy': "Exactitud (Accuracy): {accuracy}",
        'mcc': "Coeficiente MCC: {mcc}",
        'precision': "Precisión",
        'recall': "Sensibilidad",
        'f1_score': "F1-Score",
        'model_comparison': "Comparación de Modelos",
        'mcc_title': "Coeficiente de Correlación de Matthews (MCC)",
        'mcc_description': "El MCC (Matthews Correlation Coefficient) mide la calidad de la clasificación binaria, especialmente en conjuntos de datos desbalanceados. Va de -1 a 1, donde 1 es perfecto, 0 es aleatorio, y -1 indica fallo total.",
        'mcc_comparison': "Comparación de MCC entre modelos:",
        'model': "Modelo",
        'mcc_gender': "MCC (Género)",
        'mcc_usage': "MCC (Uso)",
        'mcnemar_title': "Prueba de McNemar: Comparación de Modelos",
        'mcnemar_description': "La prueba de McNemar permite comparar directamente si dos modelos tienen diferencias estadísticamente significativas en su rendimiento. Se considera significativa si p < 0.05.",
        'comparison': "Comparación: {model1} vs {model2}",
        'gender': "Género",
        'usage': "Uso",
        'significant': "(significativo)",
        'not_significant': "(no significativo)",
        'conclusion': "Conclusión General",
        'significant_differences': "Se encontraron diferencias estadísticamente significativas en las siguientes comparaciones:\n- {differences}\n\nEsto sugiere que los modelos comparados no tienen el mismo rendimiento y uno puede ser superior al otro.",
        'no_differences': "No se encontraron diferencias estadísticamente significativas (p ≥ 0.05) entre los modelos en ninguna de las comparaciones. Esto indica que su rendimiento es similar en los conjuntos evaluados.",
        'footer': "Reporte generado automáticamente por el Clasificador de Camisetas"
    },
    'en': {
        'report_title': "T-shirt Analysis Report",
        'analysis_date': "Analysis date: {date}",
        'model_used': "Model used: {model}",
        'analyzed_image': "Analyzed image:",
        'main_results': "Main results:",
        'attribute': "Attribute",
        'prediction': "Prediction",
        'confidence': "Confidence",
        'attribute_details': "Details by attribute:",
        'category': "Category",
        'probability': "Probability",
        'model_evaluation': "Model Evaluation",
        'evaluation_description': "This section shows the confusion matrices and metrics of the different models evaluated for t-shirt classification.",
        'epochs': "Epochs:",
        'training': "Training:",
        'validation': "Validation:",
        'testing': "Testing:",
        'metrics_for': "Metrics for {attribute}:",
        'accuracy': "Accuracy: {accuracy}",
        'mcc': "MCC Coefficient: {mcc}",
        'precision': "Precision",
        'recall': "Recall",
        'f1_score': "F1-Score",
        'model_comparison': "Model Comparison",
        'mcc_title': "Matthews Correlation Coefficient (MCC)",
        'mcc_description': "The MCC (Matthews Correlation Coefficient) measures the quality of binary classification, especially in unbalanced datasets. It ranges from -1 to 1, where 1 is perfect, 0 is random, and -1 indicates total failure.",
        'mcc_comparison': "MCC comparison between models:",
        'model': "Model",
        'mcc_gender': "MCC (Gender)",
        'mcc_usage': "MCC (Usage)",
        'mcnemar_title': "McNemar Test: Model Comparison",
        'mcnemar_description': "The McNemar test allows direct comparison of whether two models have statistically significant differences in their performance. It is considered significant if p < 0.05.",
        'comparison': "Comparison: {model1} vs {model2}",
        'gender': "Gender",
        'usage': "Usage",
        'significant': "(significant)",
        'not_significant': "(not significant)",
        'conclusion': "General Conclusion",
        'significant_differences': "Statistically significant differences were found in the following comparisons:\n- {differences}\n\nThis suggests that the compared models do not have the same performance and one may be superior to the other.",
        'no_differences': "No statistically significant differences (p ≥ 0.05) were found between models in any of the comparisons. This indicates that their performance is similar in the evaluated sets.",
        'footer': "Report automatically generated by the T-shirt Classifier"
    }
}

# Función para obtener traducciones
def t(key, lang='es', **kwargs):
    """Obtiene la traducción para una clave dada"""
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    try:
        return translation.format(**kwargs) if kwargs else translation
    except (KeyError, IndexError):
        # Si hay error en el formato, devolver la cadena sin formatear
        return translation

def pdf_t(key, lang='es', **kwargs):
    """Obtiene la traducción para el reporte PDF"""
    translation = PDF_TRANSLATIONS.get(lang, {}).get(key, key)
    try:
        return translation.format(**kwargs) if kwargs else translation
    except (KeyError, IndexError):
        # Si hay error en el formato, devolver la cadena sin formatear
        return translation


# Inicializa el idioma si no está en la sesión
if 'language' not in st.session_state:
    st.session_state.language = 'es'

# Configuración de banderas
FLAGS = {
    'es': {
        'url': 'https://flagcdn.com/w40/es.png',
        'tooltip': 'Español',
        'border_color': '#AA151B'
    },
    'en': {
        'url': 'https://flagcdn.com/w40/gb.png',
        'tooltip': 'English',
        'border_color': '#012169'
    }
}

# Función para formatear las opciones del radio button
def format_radio_option(lang_code):
    flag_data = FLAGS[lang_code]
    return f"![{flag_data['tooltip']}]({flag_data['url']}) {flag_data['tooltip']}"

# Mostrar en la barra lateral
with st.sidebar:
    st.markdown("### 🌍 Idioma / Language")
    
    # Crear radio button con banderas y texto
    selected_lang = st.radio(
        label="Selecciona un idioma",
        options=list(FLAGS.keys()),
        format_func=format_radio_option,
        index=list(FLAGS.keys()).index(st.session_state.language),
        key="language_selector",
        label_visibility="collapsed"
    )
    
    # Actualizar el idioma en session_state
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
    
    # Aplicar estilo CSS para alinear banderas y texto
    st.markdown(
        """
        <style>
            /* Alinear banderas y texto en el radio button */
            div[data-testid="stRadio"] > div > div {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px;
                border-radius: 5px;
            }
            /* Estilo para la bandera seleccionada */
            div[role="radiogroup"] div[aria-checked="true"] img {
                border: 2px solid;
                border-radius: 5px;
                padding: 2px;
                box-shadow: 0 0 8px rgba(0, 0, 0, 0.2);
            }
            /* Asignar colores de borde específicos */
            div[role="radiogroup"] div[aria-checked="true"]:has(img[src*="es.png"]) img {
                border-color: #AA151B;
            }
            div[role="radiogroup"] div[aria-checked="true"]:has(img[src*="gb.png"]) img {
                border-color: #012169;
            }
            /* Ajustar tamaño de la bandera */
            div[data-testid="stRadio"] img {
                width: 30px;
                vertical-align: middle;
            }
            /* Estilo para el hover */
            div[data-testid="stRadio"] > div > div:hover {
                background: rgba(0, 0, 0, 0.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Idioma seleccionado actual
language = st.session_state.language


# Cargar los resultados de evaluación
with open('metricas_modelos.json') as f:
    metricas_modelos = json.load(f)
    
# Inicialización de session_state
if 'prediccion' not in st.session_state:
    st.session_state.prediccion = None
if 'imagen' not in st.session_state:
    st.session_state.imagen = None
if 'archivo_subido' not in st.session_state:
    st.session_state.archivo_subido = None

# ======================
# CONFIGURACIÓN INICIAL
# ======================
MODEL_PATH = 'model/mobilenet_final.keras'
ATTRIBUTES = ['gender', 'usage']
IMG_SIZE = (224, 224)

# Traducciones para atributos y valores
TRADUCCION_ATRIBUTOS = {'gender': t('gender', language), 'usage': t('usage', language)}
TRADUCCION_VALORES = {
    'Men': t('men', language), 'Women': t('women', language), 
    'Casual': t('casual', language), 'Sports': t('sports', language),
    'Hombre': t('men', language), 'Mujer': t('women', language),
    'Deportivo': t('sports', language)
}

# ======================
# FUNCIONES AUXILIARES
# ======================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model(MODEL_PATH)
        return modelo
    except Exception as e:
        st.error(t('model_error', language, e=str(e)))
        return None

@st.cache_resource
def obtener_codificadores_etiquetas():
    return {
        'gender': LabelEncoder().fit(['Men', 'Women']),
        'usage': LabelEncoder().fit(['Casual', 'Sports'])
    }

def predecir_atributos_camiseta(archivo_subido, modelo, codificadores):
    try:
        img = Image.open(archivo_subido)
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predicciones = modelo.predict(img_array, verbose=0)

        resultados = {}
        for i, attr in enumerate(ATTRIBUTES):
            clase_predicha = np.argmax(predicciones[i])
            etiqueta_predicha = codificadores[attr].inverse_transform([clase_predicha])[0]
            confianza = np.max(predicciones[i])
            resultados[attr] = {
                'label': etiqueta_predicha, 
                'confidence': float(confianza),
                'probabilities': {cls: float(prob) for cls, prob in 
                                zip(codificadores[attr].classes_, predicciones[i][0])}
            }
        
        return resultados, img

    except Exception as e:
        st.error(t('processing_error', language, e=str(e)))
        return None, None

def generar_enlace_descarga_pdf(ruta_archivo, texto_boton, language):
    with open(ruta_archivo, "rb") as f:
        datos = f.read()
    bin_str = base64.b64encode(datos).decode()
    filename = f"{t('pdf_report_name', language)}.pdf"
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{filename}">{texto_boton}</a>'
    return href

def obtener_metricas_modelo(modelo, atributo):
    """Obtiene las métricas de un modelo específico para un atributo dado"""
    if modelo in metricas_modelos['models'] and atributo in metricas_modelos['models'][modelo]:
        return metricas_modelos['models'][modelo][atributo]
    return None

def generar_reporte_prediccion(prediccion, img, nombre_modelo="MobileNet"):
    """Genera un PDF con el reporte de predicción y evaluación de modelos"""
    # Crear PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, pdf_t('report_title', language), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(5)
    
    # Información general
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, pdf_t('analysis_date', language, date=datetime.now().strftime('%d/%m/%Y %H:%M:%S')), 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, pdf_t('model_used', language, model=nombre_modelo), 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Guardar imagen temporalmente
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_img.name, format='JPEG', quality=90)
    
    # Agregar imagen al PDF
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('analyzed_image', language), 0, 1)
    pdf.image(temp_img.name, x=10, w=180)
    pdf.ln(15)

    # Resultados principales
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('main_results', language), 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(70, 10, pdf_t('attribute', language), 1)
    pdf.cell(70, 10, pdf_t('prediction', language), 1)
    pdf.cell(50, 10, pdf_t('confidence', language), 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 12)
    for attr, data in prediccion.items():
        atributo = TRADUCCION_ATRIBUTOS.get(attr, attr)
        valor = TRADUCCION_VALORES.get(data['label'], data['label'])
        
        pdf.cell(70, 10, atributo, 1)
        pdf.cell(70, 10, valor, 1)
        pdf.cell(50, 10, f"{data['confidence']:.1%}", 1)
        pdf.ln()
    
    # Detalles por atributo
    pdf.ln(15)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('attribute_details', language), 0, 1)
    pdf.ln(5)
    
    for attr, data in prediccion.items():
        atributo = TRADUCCION_ATRIBUTOS.get(attr, attr)
        
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, f"{pdf_t('attribute', language)}: {atributo}", 0, 1)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", '', 12)
        pdf.cell(0, 10, f"{pdf_t('prediction', language)}: {TRADUCCION_VALORES.get(data['label'], data['label'])}", 0, 1)
        pdf.cell(0, 10, f"{pdf_t('confidence', language)}: {data['confidence']:.1%}", 0, 1)
        pdf.ln(3)
        
        # Tabla de probabilidades
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(90, 8, pdf_t('category', language), 1)
        pdf.cell(90, 8, pdf_t('probability', language), 1)
        pdf.ln()
        
        pdf.set_font("Helvetica", '', 11)
        for cls, prob in data['probabilities'].items():
            clase_traducida = TRADUCCION_VALORES.get(cls, cls)
            pdf.cell(90, 8, clase_traducida, 1)
            pdf.cell(90, 8, f"{prob:.1%}", 1)
            pdf.ln()
        
        pdf.ln(10)
    
    # ===================================
    # SECCIÓN DE EVALUACIÓN DE MODELOS
    # ===================================
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, pdf_t('model_evaluation', language), 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", '', 12)
    pdf.multi_cell(0, 10, pdf_t('evaluation_description', language))
    pdf.set_font("Helvetica", '', 12)
    pdf.ln(5)
    
    # Línea 1: Épocas
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(20, 10, pdf_t('epochs', language), 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "30", 0, 1)

    # Línea 2: Entrenamiento
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(35, 10, pdf_t('training', language), 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "6,114 (70%)", 0, 1)

    # Línea 3: Validación
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(25, 10, pdf_t('validation', language), 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "1,310 (15%)", 0, 1)

    # Línea 4: Prueba
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(20, 10, pdf_t('testing', language), 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "1,310 (15%)", 0, 1)
    pdf.ln(15)

    # Mostrar métricas para cada modelo
    for modelo in metricas_modelos['models'].keys():
        # --------------------------
        # ENCABEZADO DEL MODELO
        # --------------------------
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, modelo, 0, 1)
        pdf.ln(5)
        
        # Matriz de confusión (asumiendo que existe una imagen)
        try:
            pdf.image(f"matriz/{modelo.lower()}.png", x=10, w=180)
            pdf.ln(10)
        except:
            pass
        
        # Mostrar métricas para cada atributo
        for atributo in ['gender', 'usage']:
            metricas = obtener_metricas_modelo(modelo, atributo)
            if not metricas:
                continue
                
            nombre_atributo = TRADUCCION_ATRIBUTOS.get(atributo, atributo)
            
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, pdf_t('metrics_for', language, attribute=nombre_atributo), 0, 1)
            pdf.ln(3)
            
            # Exactitud general
            pdf.set_font("Helvetica", '', 11)
            pdf.cell(0, 8, pdf_t('accuracy', language, accuracy=f"{metricas['accuracy']:.2%}"), 0, 1)
            pdf.cell(0, 8, pdf_t('mcc', language, mcc=f"{metricas['mcc']:.4f}"), 0, 1)
            pdf.ln(5)
            
            # Tabla de métricas
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(60, 8, pdf_t('category', language), 1)
            pdf.cell(40, 8, pdf_t('precision', language), 1)
            pdf.cell(40, 8, pdf_t('recall', language), 1)
            pdf.cell(40, 8, pdf_t('f1_score', language), 1)
            pdf.ln()
            
            pdf.set_font("Helvetica", '', 11)
            
            # Obtener las categorías (Hombre/Mujer o Casual/Deportivo)
            categorias = [k for k in metricas.keys() if k not in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc']]
            
            for i, cat in enumerate(categorias):
                # Usar las métricas individuales si existen
                if isinstance(metricas[cat], dict):
                    precision = metricas[cat]['precision']
                    recall = metricas[cat]['recall']
                    f1 = metricas[cat]['f1_score']
                else:
                    # O usar las listas de métricas
                    precision = metricas['precision'][i]
                    recall = metricas['recall'][i]
                    f1 = metricas['f1_score'][i]
                
                nombre_categoria = TRADUCCION_VALORES.get(cat, cat)
                pdf.cell(60, 8, nombre_categoria, 1)
                pdf.cell(40, 8, f"{precision:.2%}", 1)
                pdf.cell(40, 8, f"{recall:.2%}", 1)
                pdf.cell(40, 8, f"{f1:.2%}", 1)
                pdf.ln()
            
            pdf.ln(10)
    
    # ===================================
    # COMPARACIÓN DE MODELOS
    # ===================================
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, pdf_t('model_comparison', language), 0, 1, 'C')
    pdf.ln(10)
    
    # --------------------------
    # COMPARACIÓN CON MCC
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('mcc_title', language), 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.multi_cell(0, 8, pdf_t('mcc_description', language))
    pdf.ln(10)
    
    # Tabla comparativa de MCC
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, pdf_t('mcc_comparison', language), 0, 1)
    pdf.ln(3)

    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(70, 8, pdf_t('model', language), 1)
    pdf.cell(60, 8, pdf_t('mcc_gender', language), 1)
    pdf.cell(60, 8, pdf_t('mcc_usage', language), 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    for modelo, datos in metricas_modelos['models'].items():
        mcc_gender = datos['gender']['mcc']
        mcc_usage = datos['usage']['mcc']
        
        pdf.cell(70, 8, modelo, 1)
        pdf.cell(60, 8, f"{mcc_gender:.4f}", 1)
        pdf.cell(60, 8, f"{mcc_usage:.4f}", 1)
        pdf.ln()
    
    pdf.ln(15)
    
    # --------------------------
    # PRUEBA DE McNEMAR
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('mcnemar_title', language), 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.multi_cell(0, 8, pdf_t('mcnemar_description', language))
    pdf.ln(10)
    
    diferencias = []  # guardamos diferencias significativas

    for comparacion, resultados in metricas_modelos['mcnemar'].items():
        modelo1, modelo2 = comparacion.split('_vs_')
        
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 8, pdf_t('comparison', language, model1=modelo1, model2=modelo2), 0, 1)
        pdf.ln(2)
        
        # Género
        chi2_gen = resultados['gender']['chi2']
        p_gen = resultados['gender']['pvalue']
        sig_gen = pdf_t('significant', language) if p_gen < 0.05 else pdf_t('not_significant', language)
        if p_gen < 0.05:
            diferencias.append(f"{modelo1} vs {modelo2} ({pdf_t('gender', language)})")

        pdf.set_font("Helvetica", '', 10)
        pdf.cell(0, 8, f"  - {pdf_t('gender', language)}: Chi2 = {chi2_gen:.4f}, p = {p_gen:.4f} {sig_gen}", 0, 1)
        
        # Uso
        chi2_uso = resultados['usage']['chi2']
        p_uso = resultados['usage']['pvalue']
        sig_uso = pdf_t('significant', language) if p_uso < 0.05 else pdf_t('not_significant', language)
        if p_uso < 0.05:
            diferencias.append(f"{modelo1} vs {modelo2} ({pdf_t('usage', language)})")

        pdf.cell(0, 8, f"  - {pdf_t('usage', language)}: Chi2 = {chi2_uso:.4f}, p = {p_uso:.4f} {sig_uso}", 0, 1)
        pdf.ln(5)

    # Interpretación final
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, pdf_t('conclusion', language), 0, 1)
    pdf.set_font("Helvetica", '', 10)
    if diferencias:
        pdf.multi_cell(0, 8, pdf_t('significant_differences', language, differences="\n- ".join(diferencias)))
    else:
        pdf.multi_cell(0, 8, pdf_t('no_differences', language))

    # Pie de página
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, pdf_t('footer', language), 0, 0, 'C')
    
    # Guardar PDF temporal
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    pdf.output(temp_pdf.name)
    
    # Limpiar archivos temporales
    try:
        os.unlink(temp_img.name)
    except:
        pass
    
    return temp_pdf.name

# ======================
# INTERFAZ PRINCIPAL
# ======================
# Cargar recursos
modelo = cargar_modelo()
codificadores = obtener_codificadores_etiquetas()

# Título de la aplicación
st.title(t('app_title', language))
st.markdown(t('app_description', language))
st.markdown(f"- **{t('gender', language)}**: {t('men', language)} / {t('women', language)}")
st.markdown(f"- **{t('usage', language)}**: {t('casual', language)} / {t('sports', language)}")

# Widget para subir archivo
archivo_subido = st.file_uploader(t('upload_label', language), 
                                type=['jpg', 'jpeg', 'png'],
                                key="subidor_archivos")

# Actualizar session_state
if archivo_subido is not None:
    st.session_state.archivo_subido = archivo_subido

# Mostrar imagen cargada si existe
if st.session_state.archivo_subido is not None:
    st.image(
        st.session_state.archivo_subido,
        caption=t('image_uploaded', language),  # Ahora se traducirá
        width=300
    )
    
    # Botón para realizar predicción
    if st.button(t('analyze_button', language)):
        with st.spinner(t('analyzing', language)):
            st.session_state.prediccion, st.session_state.imagen = predecir_atributos_camiseta(
                st.session_state.archivo_subido, modelo, codificadores)
            
        if st.session_state.prediccion:
            st.success(t('analysis_complete', language))

# Mostrar resultados si existen
if st.session_state.prediccion and st.session_state.imagen:
    # Mostrar resultados en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('analyzed_image', language))
        st.image(st.session_state.imagen, use_container_width=True)
    
    with col2:
        st.subheader(t('analysis_results', language))
    
        for attr, data in st.session_state.prediccion.items():
            nombre_atributo = TRADUCCION_ATRIBUTOS[attr]
            valor_traducido = TRADUCCION_VALORES[data['label']]
            
            # Barra de progreso para la confianza
            st.progress(data['confidence'], text=f"**{nombre_atributo}**: {valor_traducido} ({data['confidence']:.1%})")
            
            # Mostrar probabilidades en un expander
            with st.expander(t('details', language, attribute=nombre_atributo.lower())):
                for cls, prob in data['probabilities'].items():
                    clase_traducida = TRADUCCION_VALORES[cls]
                    st.metric(label=clase_traducida, value=f"{prob:.1%}")
    
    # Sección para generar reporte PDF
    st.markdown("---")
    st.subheader(t('generate_report', language))
    
    if st.button(t('generate_report', language)):
        with st.spinner(t('generating_pdf', language)):
            try:
                pdf_path = generar_reporte_prediccion(
                    st.session_state.prediccion, 
                    st.session_state.imagen,
                    nombre_modelo="MobileNet"
                )
                
                st.success(t('report_success', language))
                st.markdown(
                    generar_enlace_descarga_pdf(
                        pdf_path, 
                        t('download_report', language),
                        language
                    ), 
                    unsafe_allow_html=True
                )
                
                # Limpiar archivo temporal
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            except Exception as e:
                st.error(t('report_error', language, e=str(e)))

# Información adicional
st.sidebar.markdown(f"## {t('about_title', language)}")
st.sidebar.info(t('about_content', language))