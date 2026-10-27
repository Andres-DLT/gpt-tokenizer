"""
Tokenizador BPE (Byte Pair Encoding) a nivel de bytes, al estilo de GPT,
construido desde cero.

Tres clases, de menor a mayor complejidad:

  BasicTokenizer      BPE puro sobre los bytes UTF-8 del texto.
  RegexTokenizer      Como GPT-2/GPT-4: primero parte el texto con una regex
                      (para no fusionar letras con espacios o signos) y
                      admite tokens especiales como <|endoftext|>.
  PretrainedTokenizer Carga el vocabulario oficial de OpenAI (gpt2,
                      cl100k_base = GPT-4, o200k_base = GPT-4o) y reproduce
                      exactamente sus tokens con NUESTRO código de BPE.
                      Solo usa `tiktoken` para descargar el archivo de rangos.

Idea central del BPE:
  1. Partimos de 256 tokens, uno por cada byte posible.
  2. Contamos qué pareja de tokens consecutivos aparece más.
  3. La fusionamos en un token nuevo (256, 257, ...) y repetimos.
  Codificar = aplicar las fusiones aprendidas, en el mismo orden.
  Decodificar = concatenar los bytes de cada token y leerlos como UTF-8.
"""

import unicodedata
from collections import Counter

import regex as re  # el módulo `re` estándar no soporta \p{L}, \p{N}, ...

# Patrones de pre-tokenización. Cada trozo que encuentran se tokeniza por
# separado, así que el BPE nunca fusiona a través de sus fronteras.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def get_stats(ids, counts=None, weight=1):
    """Cuenta las parejas consecutivas: [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, idx):
    """Sustituye cada aparición de `pair` por `idx`: ([1,2,3,1,2], (1,2), 4) -> [4,3,4]."""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def render_token(t: bytes) -> str:
    """Versión imprimible de un token: escapa los caracteres de control (\\n, \\t, ...)."""
    s = t.decode("utf-8", errors="replace")
    return "".join(ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}" for ch in s)


# ---------------------------------------------------------------------------
# Clase base: vocabulario, guardar y cargar
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self):
        self.merges = {}          # (int, int) -> int, en el orden en que se aprendieron
        self.pattern = ""         # regex de pre-tokenización ("" = ninguna)
        self.special_tokens = {}  # str -> int, p. ej. {"<|endoftext|>": 100257}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        """id -> bytes. Los padres de una fusión siempre existen antes que el hijo."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Escribe dos archivos:
          <prefijo>.model  lo necesario para load()
          <prefijo>.vocab  versión legible para humanos (no se usa al cargar)
        """
        with open(file_prefix + ".model", "w", encoding="utf-8") as f:
            f.write("gptbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(file_prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    p0, p1 = inverted_merges[idx]
                    f.write(f"[{render_token(self.vocab[p0])}][{render_token(self.vocab[p1])}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges, special_tokens = {}, {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            assert f.readline().strip() == "gptbpe v1"
            self.pattern = f.readline().rstrip("\n")
            for _ in range(int(f.readline())):
                special, special_idx = f.readline().rstrip("\n").rsplit(" ", 1)
                special_tokens[special] = int(special_idx)
            for line in f:
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


# ---------------------------------------------------------------------------
# 1) BPE básico sobre bytes
# ---------------------------------------------------------------------------

class BasicTokenizer(Tokenizer):

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break  # el texto ya es un único token
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"fusión {i + 1}/{vocab_size - 256}: {pair} -> {idx} "
                      f"({render_token(vocab[idx])!r}) aparecía {stats[pair]} veces")
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        return self._encode_bytes(text.encode("utf-8"))

    def _encode_bytes(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            # De las parejas presentes, la que se aprendió primero (id más bajo).
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # ya no queda nada que fusionar
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # Un token puede cortar un carácter multibyte a la mitad: "replace" evita errores.
        return text_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 2) BPE con pre-tokenización por regex y tokens especiales (GPT-2 / GPT-4)
# ---------------------------------------------------------------------------

class RegexTokenizer(BasicTokenizer):

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {}
        self._cache = {}  # trozo de texto -> ids; los textos repiten mucho los mismos trozos

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # El BPE nunca cruza la frontera entre trozos, así que basta con
        # procesar cada trozo distinto una vez, ponderado por su frecuencia.
        chunk_counts = Counter(self.compiled_pattern.findall(text))
        chunks = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights = list(chunk_counts.values())

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(vocab_size - 256):
            stats = {}
            for ids, w in zip(chunks, weights):
                get_stats(ids, stats, w)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(ids, pair, idx) for ids in chunks]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"fusión {i + 1}/{vocab_size - 256}: {pair} -> {idx} "
                      f"({render_token(vocab[idx])!r}) aparecía {stats[pair]} veces")
        self.merges = merges
        self.vocab = vocab
        for special, special_idx in self.special_tokens.items():
            self.vocab[special_idx] = special.encode("utf-8")
        self._cache.clear()

    def register_special_tokens(self, special_tokens):
        """special_tokens: dict str -> int, p. ej. {"<|endoftext|>": 100257}."""
        for special, idx in special_tokens.items():
            if idx in self.vocab and self.vocab[idx] != special.encode("utf-8"):
                raise ValueError(f"el id {idx} de {special!r} ya lo usa otro token")
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        for special, idx in self.special_tokens.items():
            self.vocab[idx] = special.encode("utf-8")

    def load(self, model_file):
        super().load(model_file)
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self._cache.clear()

    def encode_ordinary(self, text):
        """Codifica ignorando los tokens especiales (se tratan como texto normal)."""
        ids = []
        for chunk in self.compiled_pattern.findall(text):
            chunk_ids = self._cache.get(chunk)
            if chunk_ids is None:
                chunk_ids = self._encode_bytes(chunk.encode("utf-8"))
                self._cache[chunk] = chunk_ids
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        allowed_special, igual que en tiktoken:
          "all"        todos los tokens especiales se reconocen
          "none"       ninguno: "<|endoftext|>" se codifica como texto normal
          "none_raise" ninguno, y error si aparece alguno en el texto (por defecto)
          set(...)     solo los indicados
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"el texto contiene el token especial {token!r}; "
                                     f"usa allowed_special para permitirlo")
        elif isinstance(allowed_special, (set, frozenset)):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special!r} no es válido")

        if not special:
            return self.encode_ordinary(text)

        # Partimos por los tokens especiales (los más largos primero, por si uno
        # es prefijo de otro); el grupo de captura los conserva en la lista.
        names = sorted(special, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(k) for k in names) + ")"
        ids = []
        for part in re.split(special_pattern, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def _token_bytes(self, idx):
        if idx in self.inverse_special_tokens:
            return self.inverse_special_tokens[idx].encode("utf-8")
        if idx in self.vocab:
            return self.vocab[idx]
        raise ValueError(f"id de token desconocido: {idx}")

    def decode(self, ids):
        text_bytes = b"".join(self._token_bytes(idx) for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 3) Vocabularios oficiales de OpenAI con nuestro propio BPE
# ---------------------------------------------------------------------------

def bpe(mergeable_ranks, token, max_rank):
    """Aplica BPE a `token` usando solo las fusiones de rango < max_rank."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    """
    tiktoken guarda solo token -> rango, no las fusiones. Pero podemos
    reconstruirlas: si aplicamos BPE a un token usando únicamente las fusiones
    anteriores a él, quedan exactamente las dos piezas que lo forman.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2, f"no se pudo reconstruir la fusión de {token!r}"
        merges[(mergeable_ranks[pair[0]], mergeable_ranks[pair[1]])] = rank
    return merges


class PretrainedTokenizer(RegexTokenizer):
    """
    Reproduce gpt2, cl100k_base (GPT-4/GPT-3.5) u o200k_base (GPT-4o) token a token.
    Solo usamos tiktoken para obtener la regex, los rangos y los tokens especiales.
    """

    def __init__(self, encoding_name="cl100k_base"):
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        mergeable_ranks = enc._mergeable_ranks
        super().__init__(pattern=enc._pat_str)
        self.name = encoding_name
        self.merges = recover_merges(mergeable_ranks)
        # En estos vocabularios los 256 tokens de un byte no tienen id = valor del
        # byte (en GPT-2 se ordenaron para que los imprimibles fueran primero).
        # Traducimos cada byte a su id antes del BPE y lo deshacemos al decodificar.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.vocab = self._build_vocab()
        self.register_special_tokens(enc._special_tokens)

    def _encode_bytes(self, text_bytes):
        return super()._encode_bytes(bytes(self.byte_shuffle[b] for b in text_bytes))

    def _token_bytes(self, idx):
        if idx in self.inverse_special_tokens:
            return self.inverse_special_tokens[idx].encode("utf-8")
        return bytes(self.inverse_byte_shuffle[b] for b in self.vocab[idx])

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("este tokenizador ya viene entrenado")

    def save(self, file_prefix):
        raise NotImplementedError("usa tiktoken para obtener los archivos originales")

    def load(self, model_file):
        raise NotImplementedError("usa tiktoken para obtener los archivos originales")
