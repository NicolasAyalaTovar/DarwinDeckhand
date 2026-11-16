"""Darwin Deckhand: ajuste fino (fine-tuning) de GPT-2 con textos de Charles Darwin.

Flujo:
    1. Extrae el texto de los PDF de entrenamiento y prueba.
    2. Normaliza y tokeniza el texto (NLTK).
    3. Divide el texto en bloques de longitud fija para el modelo de lenguaje.
    4. Ajusta GPT-2 (TensorFlow) y evalúa la pérdida en el conjunto de prueba.
    5. Guarda el modelo y el tokenizador ajustados.

Uso:
    python darwin_deckhand.py --train AutoBio.pdf --test Origenespecies.pdf
"""

import argparse
import re
import string
from importlib import metadata

import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

LIBRERIAS = ["transformers", "tensorflow", "nltk", "pdfminer.six"]


def mostrar_versiones() -> None:
    for libreria in LIBRERIAS:
        try:
            print(f"{libreria}: {metadata.version(libreria)}")
        except metadata.PackageNotFoundError:
            print(f"La biblioteca {libreria} no está instalada.")


def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^a-záéíóúñü \n]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def procesar_texto_de_pdf(archivo_pdf: str, archivo_tokens: str | None = None) -> str:
    """Extrae, normaliza y tokeniza el texto de un PDF. Opcionalmente lo guarda."""
    texto = normalizar_texto(extract_text(archivo_pdf))
    texto_tokens = " ".join(word_tokenize(texto))
    if archivo_tokens:
        with open(archivo_tokens, "w", encoding="utf-8") as archivo:
            archivo.write(texto_tokens)
    return texto_tokens


def preparar_dataset(tokenizador, texto: str, longitud_bloque: int = 512, batch_size: int = 8):
    """Divide el texto completo en bloques de `longitud_bloque` tokens.

    Cada bloque se usa como ejemplo de modelado de lenguaje causal: la entrada
    son los tokens [0, n-1) y la etiqueta los tokens [1, n).
    """
    ids = tokenizador(texto, return_attention_mask=False)["input_ids"]
    n_bloques = len(ids) // longitud_bloque
    if n_bloques == 0:
        raise ValueError("El texto es demasiado corto para formar un bloque completo.")
    bloques = tf.reshape(tf.constant(ids[: n_bloques * longitud_bloque]), (n_bloques, longitud_bloque))
    entradas = {"input_ids": bloques[:, :-1], "attention_mask": tf.ones_like(bloques[:, :-1])}
    etiquetas = bloques[:, 1:]
    return tf.data.Dataset.from_tensor_slices((entradas, etiquetas)).shuffle(n_bloques).batch(batch_size)


def entrenar_modelo(modelo, dataset, epochs: int = 3, learning_rate: float = 5e-5) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch + 1}/{epochs}")
        for step, (entradas, etiquetas) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = modelo(**entradas, training=True).logits
                perdida = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    etiquetas, logits, from_logits=True))
            grads = tape.gradient(perdida, modelo.trainable_variables)
            optimizer.apply_gradients(zip(grads, modelo.trainable_variables))
            if step % 50 == 0:
                print(f"  Paso {step}: pérdida = {perdida.numpy():.4f}")


def evaluar_modelo(modelo, dataset) -> float:
    perdidas = []
    for entradas, etiquetas in dataset:
        logits = modelo(**entradas, training=False).logits
        perdidas.append(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            etiquetas, logits, from_logits=True)).numpy())
    promedio = sum(perdidas) / len(perdidas)
    print(f"Pérdida promedio en prueba: {promedio:.4f} (perplejidad ≈ {tf.exp(promedio).numpy():.1f})")
    return promedio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajusta GPT-2 con textos de Charles Darwin.")
    parser.add_argument("--train", default="AutoBio.pdf", help="PDF de entrenamiento.")
    parser.add_argument("--test", default="Origenespecies.pdf", help="PDF de evaluación.")
    parser.add_argument("--output", default="mi_modelo_gpt2_ajustado", help="Directorio de salida del modelo.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5, help="Tasa de aprendizaje.")
    parser.add_argument("--block-size", type=int, default=512, help="Tokens por bloque de entrenamiento.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mostrar_versiones()
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    texto_entrenamiento = procesar_texto_de_pdf(args.train, "tokens.txt")
    texto_prueba = procesar_texto_de_pdf(args.test, "tokens1.txt")

    tokenizador = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizador.pad_token = tokenizador.eos_token
    modelo = TFGPT2LMHeadModel.from_pretrained("gpt2")

    dataset_entrenamiento = preparar_dataset(tokenizador, texto_entrenamiento, args.block_size)
    dataset_prueba = preparar_dataset(tokenizador, texto_prueba, args.block_size)

    entrenar_modelo(modelo, dataset_entrenamiento, epochs=args.epochs, learning_rate=args.lr)
    evaluar_modelo(modelo, dataset_prueba)

    modelo.save_pretrained(args.output)
    tokenizador.save_pretrained(args.output)
    print(f"Modelo guardado en '{args.output}/'")


if __name__ == "__main__":
    main()
