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
import cv2
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
    },
    'fr': {
        'app_title': "👕 Classificateur d'Attributs de T-shirts",
        'app_description': "Téléchargez une image d'un t-shirt pour analyser ses attributs:",
        'gender': "Genre",
        'usage': "Utilisation",
        'upload_label': "Sélectionnez une image de t-shirt",
        'image_uploaded': "Image téléchargée",
        'analyze_button': "Analyser l'image",
        'analyzing': "Analyse de l'image...",
        'analysis_complete': "Analyse terminée avec succès!",
        'analyzed_image': "Image Analysée",
        'analysis_results': "Résultats de l'Analyse",
        'details': "Détails sur {attribute}",
        'generate_report': "Générer un Rapport",
        'generating_pdf': "Génération du document PDF...",
        'report_success': "✅ Rapport généré avec succès!",
        'download_report': "⬇️ Télécharger le Rapport Complet",
        'pdf_report_name': "rapport_t-shirt",
        'report_error': "❌ Erreur lors de la génération du rapport: {error}",
        'about_title': "À propos de cette application",
        'about_content': """
Cet outil utilise l'intelligence artificielle pour analyser les attributs des t-shirts.

**Attributs qu'il peut identifier:**
- **Genre**: Homme / Femme
- **Utilisation**: Casual / Sport

Téléchargez une image claire d'un t-shirt pour obtenir l'analyse.
""",
        'men': "Homme",
        'women': "Femme",
        'casual': "Casual",
        'sports': "Sport",
        'processing_error': "❌ Erreur de traitement de l'image: {error}",
        'model_error': "Erreur de chargement du modèle: {error}"
    },
    'de': {
        'app_title': "👕 T-Shirt Attribut-Klassifikator",
        'app_description': "Laden Sie ein T-Shirt-Bild hoch, um seine Attribute zu analysieren:",
        'gender': "Geschlecht",
        'usage': "Verwendung",
        'upload_label': "Wählen Sie ein T-Shirt-Bild aus",
        'image_uploaded': "Bild hochgeladen",
        'analyze_button': "Bild analysieren",
        'analyzing': "Bild wird analysiert...",
        'analysis_complete': "Analyse erfolgreich abgeschlossen!",
        'analyzed_image': "Analysiertes Bild",
        'analysis_results': "Analyseergebnisse",
        'details': "Details über {attribute}",
        'generate_report': "Bericht erstellen",
        'generating_pdf': "PDF-Dokument wird erstellt...",
        'report_success': "✅ Bericht erfolgreich erstellt!",
        'download_report': "⬇️ Vollständigen Bericht herunterladen",
        'pdf_report_name': "t-shirt_bericht",
        'report_error': "❌ Fehler beim Erstellen des Berichts: {error}",
        'about_title': "Über diese App",
        'about_content': """
Dieses Tool verwendet künstliche Intelligenz, um T-Shirt-Attribute zu analysieren.

**Attribute, die es identifizieren kann:**
- **Geschlecht**: Männer / Frauen
- **Verwendung**: Casual / Sport

Laden Sie ein klares Bild eines T-Shirts hoch, um die Analyse zu erhalten.
""",
        'men': "Männer",
        'women': "Frauen",
        'casual': "Casual",
        'sports': "Sport",
        'processing_error': "❌ Fehler bei der Bildverarbeitung: {error}",
        'model_error': "Fehler beim Laden des Modells: {error}"
    },
    'zh': {
        'app_title': "👕 T恤属性分类器",
        'app_description': "上传T恤图片以分析其属性:",
        'gender': "性别",
        'usage': "用途",
        'upload_label': "选择T恤图片",
        'image_uploaded': "图片已上传",
        'analyze_button': "分析图片",
        'analyzing': "正在分析图片...",
        'analysis_complete': "分析成功完成!",
        'analyzed_image': "已分析图片",
        'analysis_results': "分析结果",
        'details': "{attribute}详情",
        'generate_report': "生成报告",
        'generating_pdf': "正在生成PDF文档...",
        'report_success': "✅ 报告生成成功!",
        'download_report': "⬇️ 下载完整报告",
        'pdf_report_name': "t恤报告",
        'report_error': "❌ 生成报告时出错: {error}",
        'about_title': "关于此应用",
        'about_content': """
此工具使用人工智能分析T恤属性。

**可识别的属性:**
- **性别**: 男 / 女
- **用途**: 休闲 / 运动

上传清晰的T恤图片以获取分析。
""",
        'men': "男",
        'women': "女",
        'casual': "休闲",
        'sports': "运动",
        'processing_error': "❌ 图片处理错误: {error}",
        'model_error': "加载模型错误: {error}"
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
        'footer': "Reporte generado automáticamente por el Clasificador de Camisetas",
        'roc_gender': "Curva ROC Género",
        'roc_usage': "Curva ROC Uso",
        'heatmap_gender': "Mapa de calor Género",
        'heatmap_usage': "Mapa de calor Uso",
        'best_model': "El mejor modelo según el MCC promedio es {model} con:",
        'avg_mcc': "- MCC promedio: {value:.4f}",
        'gender_mcc': "- MCC para Género: {value:.4f}",
        'usage_mcc': "- MCC para Uso: {value:.4f}",
        'outperforms': "Supera al segundo mejor modelo ({model}) por {value:.4f} puntos en MCC promedio.",
        'significant_diff': "Diferencias estadísticamente significativas encontradas:",
        'confirmation': "Esto confirma que el mejor modelo tiene un rendimiento significativamente diferente a los modelos comparados.",
        'no_diff_best': "No se encontraron diferencias estadísticamente significativas con otros modelos principales."
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
        'footer': "Report automatically generated by the T-shirt Classifier",
        'roc_gender': "ROC Curve Gender",
        'roc_usage': "ROC Curve Usage",
        'heatmap_gender': "Heatmap Gender",
        'heatmap_usage': "Heatmap Usage",
        'best_model': "The best model based on average MCC is {model} with:",
        'avg_mcc': "- Average MCC: {value:.4f}",
        'gender_mcc': "- Gender MCC: {value:.4f}",
        'usage_mcc': "- Usage MCC: {value:.4f}",
        'outperforms': "Outperforms the second best model ({model}) by {value:.4f} points in average MCC.",
        'significant_diff': "Statistically significant differences found:",
        'confirmation': "This confirms that the best model performs significantly different from the compared models.",
        'no_diff_best': "No statistically significant differences found with other main models."
    },
    'fr': {
        'report_title': "Rapport d'Analyse de T-shirt",
        'analysis_date': "Date d'analyse: {date}",
        'model_used': "Modèle utilisé: {model}",
        'analyzed_image': "Image analysée:",
        'main_results': "Résultats principaux:",
        'attribute': "Attribut",
        'prediction': "Prédiction",
        'confidence': "Confiance",
        'attribute_details': "Détails par attribut:",
        'category': "Catégorie",
        'probability': "Probabilité",
        'model_evaluation': "Évaluation des Modèles",
        'evaluation_description': "Cette section montre les matrices de confusion et les métriques des différents modèles évalués pour la classification des t-shirts.",
        'epochs': "Époques:",
        'training': "Entraînement:",
        'validation': "Validation:",
        'testing': "Test:",
        'metrics_for': "Métriques pour {attribute}:",
        'accuracy': "Précision: {accuracy}",
        'mcc': "Coefficient MCC: {mcc}",
        'precision': "Précision",
        'recall': "Rappel",
        'f1_score': "F1-Score",
        'model_comparison': "Comparaison des Modèles",
        'mcc_title': "Coefficient de Corrélation de Matthews (MCC)",
        'mcc_description': "Le MCC (Matthews Correlation Coefficient) mesure la qualité de la classification binaire, en particulier dans les ensembles de données déséquilibrés. Il varie de -1 à 1, où 1 est parfait, 0 est aléatoire et -1 indique un échec total.",
        'mcc_comparison': "Comparaison MCC entre modèles:",
        'model': "Modèle",
        'mcc_gender': "MCC (Genre)",
        'mcc_usage': "MCC (Utilisation)",
        'mcnemar_title': "Test de McNemar: Comparaison de Modèles",
        'mcnemar_description': "Le test de McNemar permet de comparer directement si deux modèles présentent des différences statistiquement significatives dans leurs performances. Il est considéré comme significatif si p < 0,05.",
        'comparison': "Comparaison: {model1} vs {model2}",
        'gender': "Genre",
        'usage': "Utilisation",
        'significant': "(significatif)",
        'not_significant': "(non significatif)",
        'conclusion': "Conclusion Générale",
        'significant_differences': "Des différences statistiquement significatives ont été trouvées dans les comparaisons suivantes:\n- {differences}\n\nCela suggère que les modèles comparés n'ont pas la même performance et que l'un peut être supérieur à l'autre.",
        'no_differences': "Aucune différence statistiquement significative (p ≥ 0,05) n'a été trouvée entre les modèles dans aucune des comparaisons. Cela indique que leurs performances sont similaires dans les ensembles évalués.",
        'footer': "Rapport généré automatiquement par le Classificateur de T-shirts",
        'roc_gender': "Courbe ROC Genre",
        'roc_usage': "Courbe ROC Utilisation",
        'heatmap_gender': "Carte thermique Genre",
        'heatmap_usage': "Carte thermique Utilisation",
        'best_model': "Le meilleur modèle selon le MCC moyen est {model} avec :",
        'avg_mcc': "- MCC moyen : {value:.4f}",
        'gender_mcc': "- MCC Genre : {value:.4f}",
        'usage_mcc': "- MCC Utilisation : {value:.4f}",
        'outperforms': "Surpasse le deuxième meilleur modèle ({model}) de {value:.4f} points en MCC moyen.",
        'significant_diff': "Différences statistiquement significatives trouvées :",
        'confirmation': "Cela confirme que le meilleur modèle a une performance significativement différente des modèles comparés.",
        'no_diff_best': "Aucune différence statistiquement significative trouvée avec les autres modèles principaux."
    },
    'de': {
        'report_title': "T-Shirt-Analysebericht",
        'analysis_date': "Analyse-Datum: {date}",
        'model_used': "Verwendetes Modell: {model}",
        'analyzed_image': "Analysiertes Bild:",
        'main_results': "Hauptergebnisse:",
        'attribute': "Attribut",
        'prediction': "Vorhersage",
        'confidence': "Konfidenz",
        'attribute_details': "Details nach Attribut:",
        'category': "Kategorie",
        'probability': "Wahrscheinlichkeit",
        'model_evaluation': "Modellbewertung",
        'evaluation_description': "Dieser Abschnitt zeigt die Konfusionsmatrizen und Metriken der verschiedenen für die T-Shirt-Klassifizierung bewerteten Modelle.",
        'epochs': "Epochen:",
        'training': "Training:",
        'validation': "Validierung:",
        'testing': "Test:",
        'metrics_for': "Metriken für {attribute}:",
        'accuracy': "Genauigkeit: {accuracy}",
        'mcc': "MCC-Koeffizient: {mcc}",
        'precision': "Präzision",
        'recall': "Recall",
        'f1_score': "F1-Score",
        'model_comparison': "Modellvergleich",
        'mcc_title': "Matthews Korrelationskoeffizient (MCC)",
        'mcc_description': "Der MCC (Matthews Correlation Coefficient) misst die Qualität der binären Klassifikation, insbesondere in unausgewogenen Datensätzen. Er reicht von -1 bis 1, wobei 1 perfekt, 0 zufällig und -1 ein totaler Fehler ist.",
        'mcc_comparison': "MCC-Vergleich zwischen Modellen:",
        'model': "Modell",
        'mcc_gender': "MCC (Geschlecht)",
        'mcc_usage': "MCC (Verwendung)",
        'mcnemar_title': "McNemar-Test: Modellvergleich",
        'mcnemar_description': "Der McNemar-Test ermöglicht den direkten Vergleich, ob zwei Modelle statistisch signifikante Unterschiede in ihrer Leistung aufweisen. Er gilt als signifikant, wenn p < 0,05.",
        'comparison': "Vergleich: {model1} vs {model2}",
        'gender': "Geschlecht",
        'usage': "Verwendung",
        'significant': "(signifikant)",
        'not_significant': "(nicht signifikant)",
        'conclusion': "Allgemeine Schlussfolgerung",
        'significant_differences': "In den folgenden Vergleichen wurden statistisch signifikante Unterschiede festgestellt:\n- {differences}\n\nDies deutet darauf hin, dass die verglichenen Modelle nicht die gleiche Leistung haben und eines dem anderen überlegen sein kann.",
        'no_differences': "Es wurden keine statistisch signifikanten Unterschiede (p ≥ 0,05) zwischen den Modellen in irgendeinem der Vergleiche festgestellt. Dies deutet darauf hin, dass ihre Leistung in den bewerteten Sätzen ähnlich ist.",
        'footer': "Bericht automatisch generiert vom T-Shirt-Klassifikator",
        'roc_gender': "ROC-Kurve Geschlecht",
        'roc_usage': "ROC-Kurve Verwendung",
        'heatmap_gender': "Heatmap Geschlecht",
        'heatmap_usage': "Heatmap Verwendung",
        'best_model': "Das beste Modell nach durchschnittlichem MCC ist {model} mit:",
        'avg_mcc': "- Durchschnittlicher MCC: {value:.4f}",
        'gender_mcc': "- MCC Geschlecht: {value:.4f}",
        'usage_mcc': "- MCC Verwendung: {value:.4f}",
        'outperforms': "Übertrifft das zweitbeste Modell ({model}) um {value:.4f} Punkte im durchschnittlichen MCC.",
        'significant_diff': "Statistisch signifikante Unterschiede gefunden:",
        'confirmation': "Dies bestätigt, dass das beste Modell eine signifikant andere Leistung als die Vergleichsmodelle hat.",
        'no_diff_best': "Keine statistisch signifikanten Unterschiede zu anderen Hauptmodellen gefunden."
    },
    'zh': {
        'report_title': "T恤分析报告",
        'analysis_date': "分析日期: {date}",
        'model_used': "使用的模型: {model}",
        'analyzed_image': "已分析图片:",
        'main_results': "主要结果:",
        'attribute': "属性",
        'prediction': "预测",
        'confidence': "置信度",
        'attribute_details': "按属性详细:",
        'category': "类别",
        'probability': "概率",
        'model_evaluation': "模型评估",
        'evaluation_description': "本节显示为T恤分类评估的不同模型的混淆矩阵和指标。",
        'epochs': "训练轮数:",
        'training': "训练:",
        'validation': "验证:",
        'testing': "测试:",
        'metrics_for': "{attribute}的指标:",
        'accuracy': "准确率: {accuracy}",
        'mcc': "MCC系数: {mcc}",
        'precision': "精确率",
        'recall': "召回率",
        'f1_score': "F1分数",
        'model_comparison': "模型比较",
        'mcc_title': "马修斯相关系数(MCC)",
        'mcc_description': "MCC(Matthews Correlation Coefficient)衡量二元分类的质量,特别是在不平衡数据集中。范围从-1到1,其中1是完美的,0是随机的,-1表示完全失败。",
        'mcc_comparison': "模型间MCC比较:",
        'model': "模型",
        'mcc_gender': "MCC(性别)",
        'mcc_usage': "MCC(用途)",
        'mcnemar_title': "McNemar检验:模型比较",
        'mcnemar_description': "McNemar检验可以直接比较两个模型的性能是否存在统计学上的显著差异。如果p < 0.05则认为显著。",
        'comparison': "比较: {model1} vs {model2}",
        'gender': "性别",
        'usage': "用途",
        'significant': "(显著)",
        'not_significant': "(不显著)",
        'conclusion': "总体结论",
        'significant_differences': "在以下比较中发现统计学上的显著差异:\n- {differences}\n\n这表明比较的模型不具有相同的性能,一个可能优于另一个。",
        'no_differences': "在任何比较中均未发现模型之间存在统计学上的显著差异(p ≥ 0.05)。这表明它们的性能在评估集中相似。",
        'footer': "报告由T恤分类器自动生成",
        'roc_gender': "ROC曲线 性别",
        'roc_usage': "ROC曲线 用途",
        'heatmap_gender': "热图 性别",
        'heatmap_usage': "热图 用途"
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
    },
    'fr': {
        'url': 'https://flagcdn.com/w40/fr.png',
        'tooltip': 'Français',
        'border_color': '#0055A4'
    },
    'de': {
        'url': 'https://flagcdn.com/w40/de.png',
        'tooltip': 'Deutsch',
        'border_color': '#000000'
    },
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
            div[role="radiogroup"] div[aria-checked="true"]:has(img[src*="fr.png"]) img {
                border-color: #0055A4;
            }
            div[role="radiogroup"] div[aria-checked="true"]:has(img[src*="de.png"]) img {
                border-color: #000000;
            }
            div[role="radiogroup"] div[aria-checked="true"]:has(img[src*="cn.png"]) img {
                border-color: #DE2910;
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
# AÑADE ESTO JUSTO DEBAJO:
if 'imagen_clahe' not in st.session_state:
    st.session_state.imagen_clahe = None
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
    'Deportivo': t('sports', language),
    'Homme': t('men', language), 'Femme': t('women', language),
    'Sport': t('sports', language),
    'Männer': t('men', language), 'Frauen': t('women', language),
    '男': t('men', language), '女': t('women', language),
    '运动': t('sports', language)
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
def aplicar_clahe(imagen):
    """Aplica normalización CLAHE a una imagen"""
    # Convertir a LAB color space
    lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE al canal L (luminancia)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Fusionar canales y convertir de vuelta a RGB
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return final

def predecir_atributos_camiseta(archivo_subido, modelo, codificadores):
    try:
        img = Image.open(archivo_subido)
        img = img.resize(IMG_SIZE)
        
        # Convertir a array y aplicar CLAHE
        img_array = np.array(img)
        img_clahe = aplicar_clahe(img_array)
        
        # Preprocesar para el modelo (normalizar)
        img_for_model = img_clahe / 255.0
        img_for_model = np.expand_dims(img_for_model, axis=0)

        predicciones = modelo.predict(img_for_model, verbose=0)

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
        
        # Devolver ambas imágenes (original y CLAHE)
        return resultados, img, Image.fromarray(img_clahe)

    except Exception as e:
        st.error(t('processing_error', language, e=str(e)))
        return None, None, None

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

def generar_reporte_prediccion(prediccion, img, img_clahe, nombre_modelo="MobileNet"): 
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
    temp_img_clahe = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img_clahe.save(temp_img_clahe.name, format='JPEG', quality=90)

    # Agregar imagen al PDF
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('analyzed_image', language), 0, 1)
    pdf.image(temp_img_clahe.name, x=10, w=140)
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
        valor_traducido = TRADUCCION_VALORES.get(data['label'], data['label'])
        
        pdf.cell(70, 10, atributo, 1)
        pdf.cell(70, 10, valor_traducido, 1)
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
    


    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('roc_gender', language), 0, 1)
    pdf.image(f"curva/roc_gender.png", x=10, w=150)
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('roc_usage', language), 0, 1)
    pdf.image(f"curva/roc_usage.png", x=10, w=150)
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
    
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('heatmap_gender', language), 0, 1)
    pdf.image(f"mapa/mapa_rho_genero.png", x=10, w=150)
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, pdf_t('heatmap_usage', language), 0, 1)
    pdf.image(f"mapa/mapa_rho_uso.png", x=10, w=150)
    pdf.ln(15)


    # --------------------------
    # CONCLUSIÓN MEJORADA
    # --------------------------
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, pdf_t('conclusion', language), 0, 1)
    pdf.set_font("Helvetica", '', 10)

    # Determinar el mejor modelo basado en el MCC promedio
    modelos_mcc = []
    for modelo, datos in metricas_modelos['models'].items():
        mcc_gender = datos['gender']['mcc']
        mcc_usage = datos['usage']['mcc']
        mcc_promedio = (mcc_gender + mcc_usage) / 2
        modelos_mcc.append((modelo, mcc_promedio, mcc_gender, mcc_usage))

    # Ordenar modelos por MCC promedio
    modelos_mcc.sort(key=lambda x: x[1], reverse=True)
    mejor_modelo = modelos_mcc[0][0]
    mcc_promedio_mejor = modelos_mcc[0][1]
    mcc_gender_mejor = modelos_mcc[0][2]
    mcc_usage_mejor = modelos_mcc[0][3]

    # Texto de conclusión
    conclusion_text = pdf_t('best_model', language, model=mejor_modelo) + "\n"
    conclusion_text += pdf_t('avg_mcc', language, value=mcc_promedio_mejor) + "\n"
    conclusion_text += pdf_t('gender_mcc', language, value=mcc_gender_mejor) + "\n"
    conclusion_text += pdf_t('usage_mcc', language, value=mcc_usage_mejor) + "\n\n"

    # Comparación con otros modelos
    if len(modelos_mcc) > 1:
        segundo_mejor = modelos_mcc[1][0]
        diferencia = mcc_promedio_mejor - modelos_mcc[1][1]
        conclusion_text += pdf_t('outperforms', language, model=segundo_mejor, value=diferencia) + "\n\n"

    # Diferencias significativas
    diferencias_significativas = []
    for comparacion, resultados in metricas_modelos['mcnemar'].items():
        if resultados['gender']['pvalue'] < 0.05 and mejor_modelo in comparacion:
            modelo1, modelo2 = comparacion.split('_vs_')
            if modelo1 == mejor_modelo or modelo2 == mejor_modelo:
                otro_modelo = modelo2 if modelo1 == mejor_modelo else modelo1
                diferencias_significativas.append(f"{mejor_modelo} vs {otro_modelo} ({pdf_t('gender', language)})")

        if resultados['usage']['pvalue'] < 0.05 and mejor_modelo in comparacion:
            modelo1, modelo2 = comparacion.split('_vs_')
            if modelo1 == mejor_modelo or modelo2 == mejor_modelo:
                otro_modelo = modelo2 if modelo1 == mejor_modelo else modelo1
                diferencias_significativas.append(f"{mejor_modelo} vs {otro_modelo} ({pdf_t('usage', language)})")

    if diferencias_significativas:
        conclusion_text += pdf_t('significant_diff', language) + "\n"
        for diff in diferencias_significativas:
            conclusion_text += f"- {diff}\n"
        conclusion_text += "\n" + pdf_t('confirmation', language)
    else:
        conclusion_text += pdf_t('no_diff_best', language)

    pdf.multi_cell(0, 8, conclusion_text)
    
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
            st.session_state.prediccion, st.session_state.imagen, st.session_state.imagen_clahe = predecir_atributos_camiseta(
                st.session_state.archivo_subido, modelo, codificadores)
            
        if st.session_state.prediccion:
            st.success(t('analysis_complete', language))

# Mostrar resultados si existen
if st.session_state.prediccion and st.session_state.imagen and st.session_state.imagen_clahe:
    # Mostrar resultados en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('analyzed_image', language))
        st.image(st.session_state.imagen_clahe, use_container_width=True)
    
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
                    st.session_state.imagen_clahe, 
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