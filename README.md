# gpt-tokenizer

Tokenizador BPE (*Byte Pair Encoding*) a nivel de bytes, al estilo de GPT, construido desde cero en Python.

Con los vocabularios oficiales de OpenAI produce **exactamente los mismos ids** que `tiktoken` para GPT-2 (`gpt2`), GPT-4 (`cl100k_base`) y GPT-4o (`o200k_base`).

## Cómo funciona

1. Se empieza con 256 tokens, uno por cada byte posible: cualquier texto en cualquier idioma se puede representar.
2. Se cuenta qué pareja de tokens consecutivos aparece más, se fusiona en un token nuevo (256, 257, …) y se repite.
3. **Codificar** = aplicar las fusiones en el orden en que se aprendieron. **Decodificar** = concatenar los bytes de cada token.

## Las tres clases de `tokenizer.py`

| Clase | Qué hace |
|---|---|
| `BasicTokenizer` | BPE puro sobre los bytes UTF-8. |
| `RegexTokenizer` | Como GPT-2/GPT-4: parte el texto con una expresión regular (palabras, números, signos, espacios) y solo fusiona dentro de cada trozo. Admite tokens especiales como `<|endoftext|>`. |
| `PretrainedTokenizer` | Carga el vocabulario oficial y tokeniza con nuestro propio BPE. Reconstruye las fusiones a partir de los rangos, porque el archivo de OpenAI no las guarda. `tiktoken` solo se usa para descargar ese archivo. |

## Uso

```python
from tokenizer import RegexTokenizer, PretrainedTokenizer

tok = RegexTokenizer()
tok.train(open("muestra.txt", encoding="utf-8").read(), vocab_size=512)
tok.register_special_tokens({"<|endoftext|>": 512})

ids = tok.encode("hola mundo<|endoftext|>", allowed_special="all")
print(ids, tok.decode(ids))

tok.save("mi_tokenizador")             # escribe mi_tokenizador.model y .vocab
tok2 = RegexTokenizer()
tok2.load("mi_tokenizador.model")

gpt4 = PretrainedTokenizer("cl100k_base")
print(gpt4.encode("La información del tokenizador"))
```

`allowed_special` funciona como en `tiktoken`: `"none_raise"` (por defecto, error si el texto contiene un token especial), `"none"`, `"all"` o un `set` con los permitidos.

## Instalación y pruebas

```bash
python -m venv .venv
.venv/Scripts/python -m pip install regex tiktoken   # en Linux/macOS: .venv/bin/python
.venv/Scripts/python test_tokenizer.py
.venv/Scripts/python demo.py
```

`regex` es necesario porque el módulo `re` estándar no admite `\p{L}` ni `\p{N}`. `tiktoken` solo hace falta para `PretrainedTokenizer` y para las pruebas de equivalencia con OpenAI.

## Archivos

- `tokenizer.py`: el tokenizador.
- `test_tokenizer.py`: pruebas, incluida la comparación con `tiktoken`.
- `demo.py`: entrena con `muestra.txt` y compara con GPT-4.
- `muestra.txt`: texto de ejemplo en español para entrenar.
