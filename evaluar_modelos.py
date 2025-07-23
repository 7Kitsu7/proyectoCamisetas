import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, confusion_matrix, 
                            classification_report, precision_score, 
                            recall_score, f1_score, accuracy_score)
from statsmodels.stats.contingency_tables import mcnemar
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Configuración
MODEL_PATHS = {
    'AlexNet': 'model/alexnet_final.keras',
    'MobileNet': 'model/mobilenet_final.keras',
    'ResNet-50': 'model/resnet_final.keras',
    'MobileNetV2+AlexNet': 'saved_models/hybrid_alexnet.keras',
    'MobileNetV2+ResNet50': 'saved_models/hybrid_resnet50.keras'
}
CSV_PATH = 'styles.csv'
IMAGE_DIR = 'images'
OUTPUT_JSON = 'metricas_modelos.json'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Nombres en español para las categorías
CATEGORIAS_GENERO = ['Hombre', 'Mujer']
CATEGORIAS_USO = ['Casual', 'Deportivo']

def plot_roc_curves(resultados, target='gender'):
    """Genera y guarda curvas ROC comparativas para todos los modelos"""
    plt.figure(figsize=(10, 8))
    
    # Configuración según target
    if target == 'gender':
        title = "Curvas ROC - Clasificación por Género"
        categories = CATEGORIAS_GENERO
        # Ajustar etiquetas para modelos híbridos
        for model_name in resultados:
            if 'MobileNetV2+' in model_name:
                # Invertir las etiquetas para híbridos
                for i in range(len(resultados[model_name][target]['true'])):
                    resultados[model_name][target]['true'][i] = 1 - resultados[model_name][target]['true'][i]
                    resultados[model_name][target]['pred'][i] = 1 - resultados[model_name][target]['pred'][i]
    else:
        title = "Curvas ROC - Clasificación por Uso"
        categories = CATEGORIAS_USO
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    
    for model_name, color in zip(resultados.keys(), colors):
        y_true = np.array(resultados[model_name][target]['true'])
        y_pred = np.array(resultados[model_name][target]['pred'])
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    
    filename = f"roc_curve_{target}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Curva ROC para {target} guardada en {filename}")

def load_and_prepare_data():
    """Carga y prepara los datos"""
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    df = df[df['articleType'] == 'Tshirts'].copy()
    df = df[df['gender'].isin(['Men', 'Women'])]
    df = df[df['usage'].isin(['Casual', 'Sports'])]
    
    df['image_path'] = df['id'].astype(str) + '.jpg'
    df['image_exists'] = df['image_path'].apply(lambda x: os.path.exists(os.path.join(IMAGE_DIR, x)))
    df = df[df['image_exists']].drop(columns=['image_exists'])
    
    df['gender_encoded'] = df['gender'].map({'Men': 0, 'Women': 1})
    df['usage_encoded'] = df['usage'].map({'Casual': 0, 'Sports': 1})
    
    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[['gender', 'usage']],
        random_state=RANDOM_STATE
    )
    
    print("\n📊 Distribución en conjunto de prueba:")
    print("Género:", test_df['gender'].value_counts())
    print("\nUso:", test_df['usage'].value_counts())
    
    return test_df

def preprocess_image(image_path, img_size, model_name):
    """Preprocesamiento de imágenes"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    if 'MobileNet' in model_name:
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    else:
        img = img / 255.0
    
    return img

def evaluate_model(model, test_df, img_size, model_name):
    """Evalúa el modelo y devuelve predicciones"""
    gender_true, usage_true = [], []
    gender_pred, usage_pred = [], []
    
    for _, row in test_df.iterrows():
        # Definir tamaño de imagen específico para cada modelo
        if 'MobileNetV2+AlexNet' in model_name:
            current_img_size = (224, 224)  # MobileNetV2 requiere 224x224
        elif 'AlexNet' in model_name:
            current_img_size = (227, 227)  # AlexNet original requiere 227x227
        else:
            current_img_size = (224, 224)  # Otros modelos (MobileNet, ResNet, etc.)
        
        img = preprocess_image(
            os.path.join(IMAGE_DIR, row['image_path']),
            current_img_size,
            model_name
        )
        preds = model.predict(np.array([img]), verbose=0)
        
        # Aplicar codificación invertida solo para modelos híbridos
        if 'MobileNetV2+' in model_name:
            gender_true.append(1 - row['gender_encoded'])  # Invertir para híbridos
        else:
            gender_true.append(row['gender_encoded'])  # Mantener original para otros
            
        usage_true.append(row['usage_encoded'])
        
        # Manejo diferente para modelos híbridos
        if 'MobileNetV2+' in model_name:
            # Modelos híbridos devuelven una lista con dos salidas
            gender_pred.append(1 if preds[0][0][0] > 0.5 else 0)
            usage_pred.append(1 if preds[1][0][0] > 0.5 else 0)
        elif isinstance(preds, list):
            # Modelos multi-salida estándar
            gender_pred.append(np.argmax(preds[0]))
            usage_pred.append(np.argmax(preds[1]))
        else:
            # Modelos con salidas nombradas
            gender_pred.append(np.argmax(preds['gender']))
            usage_pred.append(np.argmax(preds['usage']))
    
    return {
        'gender': {'true': gender_true, 'pred': gender_pred},
        'usage': {'true': usage_true, 'pred': usage_pred}
    }

def save_combined_confusion_matrices(model_name, gender_data, usage_data):
    """Guarda ambas matrices de confusión en una sola imagen con barras de color"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Matrices de Confusión - {model_name}', fontsize=16, y=1.02)
    
    # Determinar el orden de las etiquetas según el tipo de modelo
    if 'MobileNetV2+' in model_name:
        gender_labels = ['Mujer', 'Hombre']  # Women:0, Men:1 para híbridos
    else:
        gender_labels = ['Hombre', 'Mujer']  # Men:0, Women:1 para otros modelos
    
    usage_labels = ['Casual', 'Deportivo']  # Uso se mantiene igual para todos
    
    # Matriz de género
    cm_gender = confusion_matrix(gender_data['true'], gender_data['pred'])
    im1 = ax1.imshow(cm_gender, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Clasificación por Género', fontsize=14)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(gender_labels)
    ax1.set_yticklabels(gender_labels)
    ax1.set_ylabel('Etiqueta verdadera')
    ax1.set_xlabel('Etiqueta predicha')
    
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Cantidad', rotation=270, labelpad=15)
    
    thresh = cm_gender.max() / 2.
    for i in range(cm_gender.shape[0]):
        for j in range(cm_gender.shape[1]):
            ax1.text(j, i, format(cm_gender[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm_gender[i, j] > thresh else "black")
    
    # Matriz de uso (se mantiene igual para todos los modelos)
    cm_usage = confusion_matrix(usage_data['true'], usage_data['pred'])
    im2 = ax2.imshow(cm_usage, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Clasificación por Uso', fontsize=14)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(usage_labels)
    ax2.set_yticklabels(usage_labels)
    ax2.set_ylabel('Etiqueta verdadera')
    ax2.set_xlabel('Etiqueta predicha')
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Cantidad', rotation=270, labelpad=15)
    
    thresh = cm_usage.max() / 2.
    for i in range(cm_usage.shape[0]):
        for j in range(cm_usage.shape[1]):
            ax2.text(j, i, format(cm_usage[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm_usage[i, j] > thresh else "black")
    
    plt.tight_layout()
    filename = f"matrices_confusion_{model_name.lower().replace('+', '_plus_')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Matrices de confusión combinadas guardadas en {filename}")

def calculate_detailed_metrics(y_true, y_pred, categories):
    """Calcula métricas detalladas por categoría"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=None).tolist(),
        'recall': recall_score(y_true, y_pred, average=None).tolist(),
        'f1_score': f1_score(y_true, y_pred, average=None).tolist(),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    for i, cat in enumerate(categories):
        metrics[cat] = {
            'precision': metrics['precision'][i],
            'recall': metrics['recall'][i],
            'f1_score': metrics['f1_score'][i]
        }
    
    return metrics

def safe_mcnemar(y_true, pred1, pred2):
    """Prueba de McNemar con manejo de errores"""
    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    a = np.sum((pred1 == y_true) & (pred2 == y_true))
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    d = np.sum((pred1 != y_true) & (pred2 != y_true))

    table = np.array([[a, b], [c, d]])
    
    if (b + c) == 0:
        return {'chi2': 0.0, 'pvalue': 1.0}

    try:
        result = mcnemar(table, exact=False, correction=True)
        return {'chi2': result.statistic, 'pvalue': result.pvalue}
    except:
        return {'chi2': 0.0, 'pvalue': 1.0}

if __name__ == '__main__':
    # 1. Cargar datos
    test_df = load_and_prepare_data()
    
    # 2. Evaluar modelos
    resultados = {}
    for name, path in MODEL_PATHS.items():
        print(f"\n🚀 Evaluando {name}...")
        try:
            model = load_model(path)
            # Definir tamaño de imagen según modelo
            if 'AlexNet' in name:
                img_size = (227, 227)
            else:
                img_size = (224, 224)
                
            resultados[name] = evaluate_model(model, test_df, img_size, name)
            
            # Guardar matrices de confusión combinadas
            save_combined_confusion_matrices(
                name,
                resultados[name]['gender'],
                resultados[name]['usage']
            )
            
        except Exception as e:
            print(f"⚠ Error al evaluar {name}: {str(e)}")
            continue
    
    # 3. Calcular métricas
    metricas = {'models': {}, 'mcnemar': {}}
    
    for name, data in resultados.items():
        metricas['models'][name] = {
            'gender': calculate_detailed_metrics(
                data['gender']['true'],
                data['gender']['pred'],
                CATEGORIAS_GENERO
            ),
            'usage': calculate_detailed_metrics(
                data['usage']['true'],
                data['usage']['pred'],
                CATEGORIAS_USO
            )
        }
    
    # McNemar
    modelos = list(resultados.keys())
    for i in range(len(modelos)):
        for j in range(i+1, len(modelos)):
            m1, m2 = modelos[i], modelos[j]
            key = f"{m1}_vs_{m2}"
            
            metricas['mcnemar'][key] = {
                'gender': safe_mcnemar(
                    resultados[m1]['gender']['true'],
                    resultados[m1]['gender']['pred'],
                    resultados[m2]['gender']['pred']
                ),
                'usage': safe_mcnemar(
                    resultados[m1]['usage']['true'],
                    resultados[m1]['usage']['pred'],
                    resultados[m2]['usage']['pred']
                )
            }
    
    # 4. Guardar resultados
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Evaluación completada. Resultados guardados en {OUTPUT_JSON}")
    
    print("\n📈 Generando curvas ROC comparativas...")
    plot_roc_curves(resultados, target='gender')
    plot_roc_curves(resultados, target='usage')