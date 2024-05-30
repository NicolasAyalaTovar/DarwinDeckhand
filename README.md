# Chatbot Entrenado con Texto de Charles Darwin - Darwin Deckhand´s 

Este proyecto utiliza un modelo GPT-2 entrenado con textos de Charles Darwin para crear un chatbot. Se procesan textos en formato PDF, se normalizan, y se usan para entrenar un modelo de lenguaje.

## Requisitos

- `transformers`
- `tensorflow`
- `nltk`
- `pdfminer.six`

Puedes instalar las dependencias usando pip:

```bash
pip install transformers tensorflow nltk pdfminer.six

## Estructura del Proyecto
# main.py: Script principal que entrena el modelo y evalúa su desempeño.
# AutoBio.pdf: Documento PDF de entrenamiento.
# tokens.txt: Archivo de texto que contiene los tokens del texto de entrenamiento.
# Origenespecies.pdf: Documento PDF de prueba.
# tokens1.txt: Archivo de texto que contiene los tokens del texto de prueba.
# mi_modelo_gpt2_ajustado/: Directorio donde se guardará el modelo y el tokenizador entrenado.

#Autor
Nicolás Ayala Tovar

Guarda este contenido en un archivo llamado `README.md` en el directorio raíz de tu proyecto.
