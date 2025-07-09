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

# Configuración de la página
st.set_page_config(page_title="Clasificador de Camisetas", page_icon="👕", layout="wide")

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
MODEL_PATH = 'model/mobilenet_final.keras'  # Cambiar la ruta al modelo AlexNet
ATTRIBUTES = ['gender', 'usage']
IMG_SIZE = (224, 224)  # Cambiado de (227, 227) a (224, 224) para MobileNet

# Traducciones para la interfaz
TRADUCCION_ATRIBUTOS = {'gender': 'Género', 'usage': 'Uso'}
TRADUCCION_VALORES = {
    'Men': 'Hombre', 'Women': 'Mujer', 
    'Casual': 'Casual', 'Sports': 'Deportivo',
    'Hombre': 'Hombre', 'Mujer': 'Mujer',
    'Deportivo': 'Deportivo'
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
        st.error(f"Error al cargar el modelo: {str(e)}")
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
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        return None, None

def generar_enlace_descarga_pdf(ruta_archivo, texto_boton):
    with open(ruta_archivo, "rb") as f:
        datos = f.read()
    bin_str = base64.b64encode(datos).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="reporte_camiseta.pdf">{texto_boton}</a>'
    return href

def obtener_metricas_modelo(modelo, atributo):
    """Obtiene las métricas de un modelo específico para un atributo dado"""
    if modelo in metricas_modelos['models'] and atributo in metricas_modelos['models'][modelo]:
        return metricas_modelos['models'][modelo][atributo]
    return None

def generar_reporte_prediccion(prediccion, img, nombre_modelo="AlexNet"):
    """Genera un PDF con el reporte de predicción y evaluación de modelos"""
    # Crear PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Reporte de Análisis de Camiseta", 0, 1, 'C')
    pdf.ln(5)
    
    # Información general
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, f"Fecha del análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Modelo utilizado: {nombre_modelo}", 0, 1)
    pdf.ln(10)
    
    # Guardar imagen temporalmente
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_img.name, format='JPEG', quality=90)
    
    # Agregar imagen al PDF
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Imagen analizada:", 0, 1)
    pdf.image(temp_img.name, x=10, w=180)
    pdf.ln(15)

    # Resultados principales
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Resultados principales:", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(70, 10, "Atributo", 1)
    pdf.cell(70, 10, "Predicción", 1)
    pdf.cell(50, 10, "Confianza", 1)
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
    pdf.cell(0, 10, "Detalles por atributo:", 0, 1)
    pdf.ln(5)
    
    for attr, data in prediccion.items():
        atributo = TRADUCCION_ATRIBUTOS.get(attr, attr)
        
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, f"Atributo: {atributo}", 0, 1)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", '', 12)
        pdf.cell(0, 10, f"Predicción: {TRADUCCION_VALORES.get(data['label'], data['label'])}", 0, 1)
        pdf.cell(0, 10, f"Confianza: {data['confidence']:.1%}", 0, 1)
        pdf.ln(3)
        
        # Tabla de probabilidades
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(90, 8, "Categoría", 1)
        pdf.cell(90, 8, "Probabilidad", 1)
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
    pdf.cell(0, 10, "Evaluación de Modelos", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", '', 12)
    pdf.multi_cell(0, 10, "Esta sección muestra las matrices de confusión y métricas de los diferentes modelos evaluados para la clasificación de camisetas.")
    pdf.set_font("Helvetica", '', 12)
    pdf.ln(5)
    
    # Línea 1: Épocas
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(20, 10, "Épocas:", 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "30", 0, 1)

    # Línea 2: Entrenamiento
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(35, 10, "Entrenamiento:", 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "6,114 imágenes (70%)", 0, 1)

    # Línea 3: Validación
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(25, 10, "Validación:", 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "1,310 imágenes (15%)", 0, 1)

    # Línea 4: Prueba
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(20, 10, "Prueba:", 0, 0)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, "1,310 imágenes (15%)", 0, 1)
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
            pdf.cell(0, 10, f"Métricas para {nombre_atributo}:", 0, 1)
            pdf.ln(3)
            
            # Exactitud general
            pdf.set_font("Helvetica", '', 11)
            pdf.cell(0, 8, f"Exactitud (Accuracy): {metricas['accuracy']:.2%}", 0, 1)
            pdf.cell(0, 8, f"Coeficiente MCC: {metricas['mcc']:.4f}", 0, 1)
            pdf.ln(5)
            
            # Tabla de métricas
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(60, 8, "Categoría", 1)
            pdf.cell(40, 8, "Precisión", 1)
            pdf.cell(40, 8, "Sensibilidad", 1)
            pdf.cell(40, 8, "F1-Score", 1)
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
    pdf.cell(0, 10, "Comparación de Modelos", 0, 1, 'C')
    pdf.ln(10)
    
    # --------------------------
    # COMPARACIÓN CON MCC
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Coeficiente de Correlación de Matthews (MCC)", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.multi_cell(0, 8, "El MCC (Matthews Correlation Coefficient) mide la calidad de la clasificación binaria, especialmente en conjuntos de datos desbalanceados. Va de -1 a 1, donde 1 es perfecto, 0 es aleatorio, y -1 indica fallo total.")
    pdf.ln(10)
    
    # Tabla comparativa de MCC
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Comparación de MCC entre modelos:", 0, 1)
    pdf.ln(3)

    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(70, 8, "Modelo", 1)
    pdf.cell(60, 8, "MCC (Género)", 1)
    pdf.cell(60, 8, "MCC (Uso)", 1)
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
    pdf.cell(0, 10, "Prueba de McNemar: Comparación de Modelos", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", '', 10)
    pdf.multi_cell(0, 8, "La prueba de McNemar permite comparar directamente si dos modelos tienen diferencias estadísticamente significativas en su rendimiento. Se considera significativa si p < 0.05.")
    pdf.ln(10)
    
    diferencias = []  # guardamos diferencias significativas

    for comparacion, resultados in metricas_modelos['mcnemar'].items():
        modelo1, modelo2 = comparacion.split('_vs_')
        
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(0, 8, f"Comparación: {modelo1} vs {modelo2}", 0, 1)
        pdf.ln(2)
        
        # Género
        chi2_gen = resultados['gender']['chi2']
        p_gen = resultados['gender']['pvalue']
        sig_gen = "(significativo)" if p_gen < 0.05 else "(no significativo)"
        if p_gen < 0.05:
            diferencias.append(f"{modelo1} vs {modelo2} (género)")

        pdf.set_font("Helvetica", '', 10)
        pdf.cell(0, 8, f"  - Género: Chi2 = {chi2_gen:.4f}, p = {p_gen:.4f} {sig_gen}", 0, 1)
        
        # Uso
        chi2_uso = resultados['usage']['chi2']
        p_uso = resultados['usage']['pvalue']
        sig_uso = "(significativo)" if p_uso < 0.05 else "(no significativo)"
        if p_uso < 0.05:
            diferencias.append(f"{modelo1} vs {modelo2} (uso)")

        pdf.cell(0, 8, f"  - Uso: Chi2 = {chi2_uso:.4f}, p = {p_uso:.4f} {sig_uso}", 0, 1)
        pdf.ln(5)

    # Interpretación final
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Conclusión General", 0, 1)
    pdf.set_font("Helvetica", '', 10)
    if diferencias:
        pdf.multi_cell(0, 8, "Se encontraron diferencias estadísticamente significativas en las siguientes comparaciones:\n- " + "\n- ".join(diferencias) + "\n\nEsto sugiere que los modelos comparados no tienen el mismo rendimiento y uno puede ser superior al otro.")
    else:
        pdf.multi_cell(0, 8, "No se encontraron diferencias estadísticamente significativas (p ≥ 0.05) entre los modelos en ninguna de las comparaciones. Esto indica que su rendimiento es similar en los conjuntos evaluados.")

    # Pie de página
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, "Reporte generado automáticamente por el Clasificador de Camisetas", 0, 0, 'C')
    
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
st.title("👕 Clasificador de Atributos de Camisetas")
st.markdown("""
Suba una imagen de una camiseta para analizar sus atributos:
- **Género**: Hombre / Mujer
- **Uso**: Casual / Deportivo
""")

# Widget para subir archivo
archivo_subido = st.file_uploader("Seleccione una imagen de camiseta", 
                                type=['jpg', 'jpeg', 'png'],
                                key="subidor_archivos")

# Actualizar session_state
if archivo_subido is not None:
    st.session_state.archivo_subido = archivo_subido

# Mostrar imagen cargada si existe
if st.session_state.archivo_subido is not None:
    st.image(st.session_state.archivo_subido, caption="Imagen cargada", width=300)
    
    # Botón para realizar predicción
    if st.button("Analizar imagen"):
        with st.spinner("Analizando imagen..."):
            st.session_state.prediccion, st.session_state.imagen = predecir_atributos_camiseta(
                st.session_state.archivo_subido, modelo, codificadores)
            
        if st.session_state.prediccion:
            st.success("¡Análisis completado con éxito!")

# Mostrar resultados si existen
if st.session_state.prediccion and st.session_state.imagen:
    # Mostrar resultados en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Analizada")
        st.image(st.session_state.imagen, use_container_width=True)
    
    with col2:
        st.subheader("Resultados del Análisis")
    
        for attr, data in st.session_state.prediccion.items():
            nombre_atributo = TRADUCCION_ATRIBUTOS[attr]
            valor_traducido = TRADUCCION_VALORES[data['label']]
            
            # Barra de progreso para la confianza
            st.progress(data['confidence'], text=f"**{nombre_atributo}**: {valor_traducido} ({data['confidence']:.1%})")
            
            # Mostrar probabilidades en un expander
            with st.expander(f"Detalles de {nombre_atributo.lower()}"):
                for cls, prob in data['probabilities'].items():
                    clase_traducida = TRADUCCION_VALORES[cls]
                    st.metric(label=clase_traducida, value=f"{prob:.1%}")
    
    # Sección para generar reporte PDF
    st.markdown("---")
    st.subheader("Generar Reporte")
    
    if st.button("Generar Reporte en PDF"):
        with st.spinner("Generando documento PDF..."):
            try:
                pdf_path = generar_reporte_prediccion(
                    st.session_state.prediccion, 
                    st.session_state.imagen,
                    nombre_modelo="MobileNet"  # Cambiado de AlexNet a MobileNet
                )
                
                st.success("✅ Reporte generado con éxito!")
                st.markdown(generar_enlace_descarga_pdf(pdf_path, "⬇️ Descargar Reporte Completo"), 
                          unsafe_allow_html=True)
                
                # Limpiar archivo temporal
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            except Exception as e:
                st.error(f"❌ Error al generar el reporte: {str(e)}")

# Información adicional
st.sidebar.markdown("## Acerca de esta aplicación")
st.sidebar.info("""
Esta herramienta utiliza inteligencia artificial para analizar atributos de camisetas.

**Atributos que puede identificar:**
- **Género**: Hombre / Mujer
- **Uso**: Casual / Deportivo

Suba una imagen clara de una camiseta para obtener el análisis.
""")