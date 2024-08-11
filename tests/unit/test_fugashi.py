# test_fugashi.py

import fugashi

tokenizer = fugashi.Tagger()
text = "これは日本語のテキストです。"
tokens = [token.surface for token in tokenizer(text)]
print(tokens)
