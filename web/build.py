"""
Prepara la web estática en una carpeta (por defecto _site/):
la página, el tokenizador, el texto de muestra y el vocabulario de GPT-4.

    python web/build.py _site
"""

import base64
import json
import os
import shutil
import sys

import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "web", "index.html"), out_dir)
    for name in ("tokenizer.py", "muestra.txt"):
        shutil.copy(os.path.join(ROOT, name), out_dir)

    # El navegador no puede descargar el vocabulario desde los servidores de
    # OpenAI (no permiten CORS), así que lo servimos junto a la página.
    enc = tiktoken.get_encoding("cl100k_base")
    with open(os.path.join(out_dir, "cl100k_base.tiktoken"), "w", encoding="ascii", newline="\n") as f:
        for token, rank in enc._mergeable_ranks.items():
            f.write(f"{base64.b64encode(token).decode()} {rank}\n")
    with open(os.path.join(out_dir, "cl100k_base.json"), "w", encoding="utf-8") as f:
        json.dump({"pattern": enc._pat_str, "special_tokens": enc._special_tokens}, f)
    print(f"web lista en {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_site")
