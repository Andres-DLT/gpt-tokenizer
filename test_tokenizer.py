import os
import tempfile
import unittest

from tokenizer import BasicTokenizer, RegexTokenizer, PretrainedTokenizer

try:
    import tiktoken
except ImportError:
    tiktoken = None

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "muestra.txt"), encoding="utf-8") as f:
    SAMPLE = f.read()

TEXTS = [
    "",
    "?",
    "hola mundo",
    "¡Hola, señor Muñoz! ¿Qué tal el pingüino?",
    "日本語のテキスト 和 中文",
    "emojis 🤗🎉 y más 👩‍👩‍👧",
    "números 1234567890 y 3.14159",
    "  espacios   múltiples\n\n\tcon tabuladores\r\n",
    "It's we'll they've I'M DON'T",
    SAMPLE,
]


class TestBasics(unittest.TestCase):

    def test_wikipedia_example(self):
        # https://en.wikipedia.org/wiki/Byte_pair_encoding
        tok = BasicTokenizer()
        tok.train("aaabdaaabac", 256 + 3)
        ids = tok.encode("aaabdaaabac")
        self.assertEqual(ids, [258, 100, 258, 97, 99])
        self.assertEqual(tok.decode(ids), "aaabdaaabac")

    def test_roundtrip(self):
        for cls in (BasicTokenizer, RegexTokenizer):
            tok = cls()
            tok.train(SAMPLE, 256 + 64)
            for text in TEXTS:
                with self.subTest(cls=cls.__name__, text=text[:20]):
                    self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_untrained_is_bytes(self):
        tok = RegexTokenizer()
        self.assertEqual(tok.encode("añ"), list("añ".encode("utf-8")))

    def test_compresses(self):
        tok = RegexTokenizer()
        tok.train(SAMPLE, 256 + 200)
        n_bytes = len(SAMPLE.encode("utf-8"))
        self.assertLess(len(tok.encode(SAMPLE)), n_bytes * 0.6)

    def test_regex_blocks_cross_chunk_merges(self):
        tok = RegexTokenizer()
        tok.train(SAMPLE, 256 + 300)
        for token in tok.vocab.values():
            # Ningún token aprendido mezcla una letra con un signo de puntuación final.
            self.assertFalse(token.rstrip(b".,!?").isalpha() and token[-1:] in (b".", b",") and len(token) > 1,
                             token)

    def test_save_load(self):
        tok = RegexTokenizer()
        tok.train(SAMPLE, 256 + 100)
        tok.register_special_tokens({"<|endoftext|>": 356, "<|fin|>": 357})
        with tempfile.TemporaryDirectory() as d:
            prefix = os.path.join(d, "tok")
            tok.save(prefix)
            tok2 = RegexTokenizer()
            tok2.load(prefix + ".model")
        self.assertEqual(tok2.merges, tok.merges)
        self.assertEqual(tok2.special_tokens, tok.special_tokens)
        text = "hola<|endoftext|>adiós<|fin|>"
        self.assertEqual(tok2.encode(text, "all"), tok.encode(text, "all"))
        self.assertEqual(tok2.decode(tok2.encode(text, "all")), text)

    def test_special_tokens(self):
        tok = RegexTokenizer()
        tok.train(SAMPLE, 256 + 50)
        tok.register_special_tokens({"<|endoftext|>": 1000})
        text = "fin<|endoftext|>inicio"
        ids = tok.encode(text, allowed_special="all")
        self.assertIn(1000, ids)
        self.assertEqual(tok.decode(ids), text)
        self.assertNotIn(1000, tok.encode(text, allowed_special="none"))
        with self.assertRaises(ValueError):
            tok.encode(text)  # por defecto, error


@unittest.skipIf(tiktoken is None, "tiktoken no está instalado")
class TestMatchesOpenAI(unittest.TestCase):
    """Nuestro BPE con los vocabularios oficiales debe dar los mismos ids que tiktoken."""

    ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]

    @classmethod
    def setUpClass(cls):
        cls.pairs = [(PretrainedTokenizer(name), tiktoken.get_encoding(name)) for name in cls.ENCODINGS]

    def test_same_ids(self):
        for ours, ref in self.pairs:
            for text in TEXTS:
                with self.subTest(encoding=ours.name, text=text[:20]):
                    ids = ours.encode(text, allowed_special="none")
                    self.assertEqual(ids, ref.encode(text, disallowed_special=()))
                    self.assertEqual(ours.decode(ids), text)

    def test_special(self):
        for ours, ref in self.pairs:
            text = "Hola<|endoftext|>mundo"
            with self.subTest(encoding=ours.name):
                self.assertEqual(ours.encode(text, "all"), ref.encode(text, allowed_special="all"))
                self.assertEqual(ours.decode(ours.encode(text, "all")), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
