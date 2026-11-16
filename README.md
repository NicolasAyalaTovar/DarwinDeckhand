# 🧭 Darwin Deckhand

**Modelo de lenguaje GPT-2 ajustado con los textos de Charles Darwin.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-GPT--2-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Darwin Deckhand convierte obras de Charles Darwin en PDF en un corpus limpio y realiza *fine-tuning* de GPT-2 para que el modelo genere texto con el estilo y el vocabulario del naturalista. Es la base para un asistente conversacional sobre historia natural y evolución.

## 🔄 Pipeline

```
PDF ──► extracción (pdfminer) ──► normalización ──► tokenización (NLTK)
    ──► bloques de 512 tokens ──► fine-tuning GPT-2 ──► evaluación (pérdida / perplejidad)
    ──► modelo guardado
```

1. **Extracción** del texto de los PDF con `pdfminer.six`.
2. **Normalización**: minúsculas, sin puntuación ni números, espacios unificados.
3. **Segmentación** del corpus completo en bloques de longitud fija, para que el modelo aprenda de *todo* el texto.
4. **Entrenamiento** con un bucle personalizado de TensorFlow (`GradientTape`) y Adam.
5. **Evaluación** sobre una obra distinta, informando pérdida y perplejidad.

## 🚀 Instalación

```bash
git clone https://github.com/NicolasAyalaTovar/DarwinDeckhand.git
cd DarwinDeckhand
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

## ▶️ Uso

Los PDF no se incluyen en el repositorio. Puedes obtener las obras de Darwin (dominio público) en [Project Gutenberg](https://www.gutenberg.org/ebooks/author/485) o [Darwin Online](http://darwin-online.org.uk/).

```bash
python darwin_deckhand.py --train AutoBio.pdf --test Origenespecies.pdf
```

| Opción | Descripción | Por defecto |
|---|---|---|
| `--train` | PDF de entrenamiento | `AutoBio.pdf` |
| `--test` | PDF de evaluación | `Origenespecies.pdf` |
| `--output` | Carpeta donde se guarda el modelo | `mi_modelo_gpt2_ajustado` |
| `--epochs` | Épocas de entrenamiento | `3` |
| `--lr` | Tasa de aprendizaje | `5e-5` |
| `--block-size` | Tokens por bloque | `512` |

### Generar texto con el modelo ajustado

```python
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

ruta = "mi_modelo_gpt2_ajustado"
tokenizer = GPT2Tokenizer.from_pretrained(ruta)
model = TFGPT2LMHeadModel.from_pretrained(ruta)

entrada = tokenizer("natural selection", return_tensors="tf")
salida = model.generate(**entrada, max_length=80, do_sample=True, top_p=0.95)
print(tokenizer.decode(salida[0], skip_special_tokens=True))
```

## 📁 Estructura

```
DarwinDeckhand/
├── darwin_deckhand.py   # Pipeline de preprocesamiento, entrenamiento y evaluación
├── requirements.txt
└── .gitignore           # Excluye PDF, tokens intermedios y el modelo entrenado
```

Archivos generados al ejecutar: `tokens.txt`, `tokens1.txt` y la carpeta del modelo.

## 🔭 Próximos pasos

- Interfaz de chat (CLI o web) sobre el modelo ajustado.
- Conservar puntuación en la normalización para mejorar la fluidez del texto generado.
- Publicar el modelo en Hugging Face Hub.

## 📄 Licencia

Distribuido bajo la licencia MIT. Consulta [`LICENSE`](LICENSE) para más información.

## 👤 Autor

**Nicolás Ayala Tovar** · [GitHub](https://github.com/NicolasAyalaTovar)
