"""Entrena un tokenizador sobre muestra.txt y lo compara con el de GPT-4."""

import os
import sys
import time

from tokenizer import BasicTokenizer, RegexTokenizer, PretrainedTokenizer, render_token

sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows no siempre usa UTF-8

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "muestra.txt"), encoding="utf-8") as f:
    text = f.read()

FRASE = "La información del tokenizador: ¡321 fusiones en 2026!"


def show(tok, s):
    ids = tok.encode(s, allowed_special="all") if isinstance(tok, RegexTokenizer) else tok.encode(s)
    pieces = [render_token(tok._token_bytes(i) if hasattr(tok, "_token_bytes") else tok.vocab[i]) for i in ids]
    print(f"  {len(ids)} tokens: " + " | ".join(pieces))
    print(f"  ids: {ids}")


os.makedirs(os.path.join(HERE, "modelos"), exist_ok=True)
n_bytes = len(text.encode("utf-8"))
print(f"Corpus: {n_bytes} bytes\n")

for cls in (BasicTokenizer, RegexTokenizer):
    tok = cls()
    t0 = time.time()
    tok.train(text, vocab_size=512)
    n = len(tok.encode(text, "all") if cls is RegexTokenizer else tok.encode(text))
    print(f"{cls.__name__}: entrenado en {time.time() - t0:.2f}s, "
          f"{n_bytes} bytes -> {n} tokens (x{n_bytes / n:.2f})")
    tok.save(os.path.join(HERE, "modelos", cls.__name__.lower()))
    show(tok, FRASE)
    print()

tok.register_special_tokens({"<|endoftext|>": 512})
print("Con token especial:")
show(tok, "fin del documento<|endoftext|>siguiente")
print()

print("Las 15 primeras fusiones aprendidas por RegexTokenizer:")
for (p0, p1), idx in list(tok.merges.items())[:15]:
    print(f"  {idx}: [{render_token(tok.vocab[p0])}] + [{render_token(tok.vocab[p1])}]"
          f" -> [{render_token(tok.vocab[idx])}]")
print()

try:
    t0 = time.time()
    gpt4 = PretrainedTokenizer("cl100k_base")
    print(f"GPT-4 (cl100k_base) con nuestro BPE, cargado en {time.time() - t0:.1f}s:")
    show(gpt4, FRASE)
    print(f"  el corpus entero: {len(gpt4.encode(text, 'none'))} tokens")
except ImportError:
    print("Instala tiktoken para comparar con el tokenizador oficial de GPT-4.")
