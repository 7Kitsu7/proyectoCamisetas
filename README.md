# 👕 Clasificación de Camisetas con Redes Neuronales Convolucionales (CNN)

![Python](https://img.shields.io/badge/Python-3.11.3-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Estado](https://img.shields.io/badge/Estado-Finalizado-brightgreen)

Este proyecto implementa un sistema automatizado de clasificación de camisetas mediante **redes neuronales convolucionales (CNN)**. Está enfocado en mejorar la **categorización multiatributo** de productos textiles en el **comercio electrónico**, utilizando las arquitecturas **AlexNet, MobileNet y ResNet-50**.

El sistema fue entrenado y evaluado sobre el dataset oficial de Kaggle:  
🔗 [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

---

## 📌 Características principales

- ✅ Clasificación de camisetas por imagen
- 🧠 Modelos CNN: AlexNet, MobileNet y ResNet-50
- 📊 Evaluación con precisión, prueba de McNemar y MCC
- 🎯 Clasificación por **género** y por **uso**
- 🌐 Interfaz gráfica amigable desarrollada con **Streamlit**
- 📦 Código abierto y reproducible para uso académico

---

## 🛠️ Tecnologías usadas

- Python 3.11.3
- Visual Studio Code (entorno de desarrollo)
- TensorFlow / Keras
- OpenCV / NumPy / Matplotlib
- scikit-learn
- Streamlit
- Sistema operativo: **Windows 11**

---

## ⚙️ Requisitos del sistema

- Python 3.11.3
- Windows 10 u 11 (también compatible con Linux/Mac)
- Paquetes del archivo `requirements.txt` (ver abajo)

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 🚀 ¿Cómo ejecutar el proyecto?

Sigue estos pasos para correr el software:

🔗 [Modelos](https://drive.google.com/drive/folders/1cP5LhZmIPMIS12xyV92dohvWlkdaXoUv)

```bash
# 1. Clonar el repositorio
git clone https://github.com/7Kitsu7/proyectoCamisetas.git
cd proyectoCamisetas

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar
Carpeta model del link "Modelos" y pegar en la raiz del proyecto

# 4. Agregar
Carpeta images del dataset "Fashion Product Images Dataset" en la raiz del proyecto

# 5. Ejecutar la interfaz gráfica
streamlit run app.py

# 6. Opcional (Si desea volver a evaluar los modelos)
streamlit run evaluar_modelos.py
```

Esto abrirá automáticamente una ventana del navegador con la interfaz del sistema, donde podrás cargar imágenes y obtener la predicción de género y uso para camisetas.

---

## 🧪 Ejemplo de uso

A continuación, se muestra la interfaz del sistema ejecutándose en Streamlit:

<p align="center"> <img src="img/interfaz1.png" width="700"/> </p> <p align="center"> <img src="img/interfaz2.png" width="700"/> </p> 
<p align="center"> <img src="img/interfaz3.png" width="700"/> </p> <p align="center"> <img src="img/interfaz4.png" width="700"/> </p>

---

## 📝 Referencias

[1] Chugh, P., & Jain, V. (2024). Artificial Intelligence Empowerment in E-Commerce: A Bibliometric Voyage. Journal of Contemporary Retail and E-Business, advance online publication. https://doi.org/10.1177/09711023241303621

[2] Ķempele, S. (2023). 20 T-Shirt industry statistics and trends. Printful Blog. https://www.printful.com/blog/t-shirt-industry-statistics

[3] Archana, R., & Jeevaraj, P. E. (2024). Deep learning models for digital image processing: a review. Artificial Intelligence Review, 57(1), 11. https://doi.org/10.1007/s10462-023-10631-z

[4] Abbas, W., Zhang, Z., Asim, M., Chen, J., & Ahmad, S. (2024). Ai-driven precision clothing classification: Revolutionizing online fashion retailing with hybrid two-objective learning. Information, 15(4), 196. https://doi.org/10.3390/info15040196 

[5] Guo, S., Huang, W., Zhang, X., Srikhanta, P., Cui, Y., Li, Y., ... & Belongie, S. (2019). The imaterialist fashion attribute dataset. In Proceedings of the IEEE/CVF international conference on computer vision workshops (pp. 0-0). https://doi.org/10.48550/arXiv.1906.05750  

[6] Zhang, H., Wu, C., … Li, M. & Smola, A. (2020). ResNeSt: Split-Attention Networks. arXiv. 
https://doi.org/10.48550/arXiv.2004.08955

[7] Abd Alaziz, H. M., Elmannai, H., Saleh, H., Hadjouni, M., Anter, A. M., Koura, A., & Kayed, M. (2023). Enhancing fashion classification with vision transformer (ViT) and developing recommendation fashion systems using DINOVA2. Electronics, 12(20), 4263. https://doi.org/10.3390/electronics12204263

[8] Liang, J., Liu, Y., & Vlassov, V. (2023). The impact of background removal on performance of neural networks for fashion image classification and segmentation. arXiv. https://doi.org/10.48550/arXiv.2308.09764

[9] Reddi, V. J., Cheng, C., Kanter, D., …, & Zhong, A. (2019). MLPerf Inference Benchmark. arXiv. https://doi.org/10.48550/arXiv.1911.02549

[10] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444. https://doi.org/10.1038/nature14539  
