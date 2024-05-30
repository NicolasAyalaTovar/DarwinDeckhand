import importlib_metadata
import tensorflow as tf
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from tensorflow.keras.optimizers import Adam
from pdfminer.high_level import extract_text

# Verificar versiones de librerías
librerias = ['transformers', 'tensorflow', 'nltk', 'pdfminer.six']
for libreria in librerias:
    try:
        version = importlib_metadata.version(libreria)
        print(f"{libreria}: {version}")
    except importlib_metadata.PackageNotFoundError:
        print(f"La biblioteca {libreria} no está instalada.")

# Descargar datos necesarios de nltk
nltk.download('punkt')

def extraer_texto_de_pdf(archivo_pdf: str) -> str:
    return extract_text(archivo_pdf)

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^a-záéíóúñü \n]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def procesar_texto_de_pdf(archivo_pdf: str, archivo_tokens: str) -> str:
    texto = extraer_texto_de_pdf(archivo_pdf)
    texto_normalizado = normalizar_texto(texto)
    tokens = word_tokenize(texto_normalizado)
    texto_tokens = ' '.join(tokens)
    with open(archivo_tokens, 'w', encoding='utf-8') as archivo:
        archivo.write(texto_tokens)
    return texto_tokens

def preparar_dataset(tokenizador, texto_de_tokens: str):
    entradas = tokenizador(texto_de_tokens, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
    etiquetas = entradas["input_ids"][:, 1:]
    entradas["input_ids"] = entradas["input_ids"][:, :-1]
    entradas["attention_mask"] = entradas["attention_mask"][:, :-1]
    dataset = tf.data.Dataset.from_tensor_slices((entradas, etiquetas))
    dataset = dataset.batch(8)
    return dataset

def entrenar_modelo(modelo, dataset, epochs=3, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    for epoch in range(epochs):
        print("\nInicio de la época", epoch)
        for step, (batch_inputs, batch_labels) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = modelo(input_ids=batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"], training=True).logits
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)

            grads = tape.gradient(loss_value, modelo.trainable_variables)
            optimizer.apply_gradients(zip(grads, modelo.trainable_variables))

            if step % 200 == 0:
                loss_scalar = tf.reduce_mean(loss_value).numpy()
                print("Pérdida en el paso {}: {:.4f}".format(step, loss_scalar))

def evaluar_modelo(modelo, dataset):
    total_loss = 0
    num_batches = 0
    for batch_inputs, batch_labels in dataset:
        logits = modelo(batch_inputs, training=False).logits
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, logits, from_logits=True)
        total_loss += tf.reduce_mean(loss_value).numpy()
        num_batches += 1
    average_loss = total_loss / num_batches
    print("Pérdida promedio en el conjunto de prueba:", average_loss)

def guardar_modelo_y_tokenizador(modelo, tokenizador, directorio_modelo):
    modelo.save_pretrained(directorio_modelo)
    tokenizador.save_pretrained(directorio_modelo)

# Principal
if __name__ == "__main__":
    ruta_pdf_entrenamiento = 'AutoBio.pdf'
    ruta_pdf_prueba = 'Origenespecies.pdf'
    archivo_tokens_entrenamiento = 'tokens.txt'
    archivo_tokens_prueba = 'tokens1.txt'

    texto_tokens_entrenamiento = procesar_texto_de_pdf(ruta_pdf_entrenamiento, archivo_tokens_entrenamiento)
    texto_tokens_prueba = procesar_texto_de_pdf(ruta_pdf_prueba, archivo_tokens_prueba)

    tokenizador = GPT2Tokenizer.from_pretrained("gpt2")
    modelo = TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizador.pad_token = tokenizador.eos_token

    dataset_entrenamiento = preparar_dataset(tokenizador, texto_tokens_entrenamiento)
    dataset_prueba = preparar_dataset(tokenizador, texto_tokens_prueba)

    entrenar_modelo(modelo, dataset_entrenamiento)
    evaluar_modelo(modelo, dataset_prueba)

    guardar_modelo_y_tokenizador(modelo, tokenizador, "mi_modelo_gpt2_ajustado")
