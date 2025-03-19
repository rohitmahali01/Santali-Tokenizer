#Load The Tokenizer
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename="./santali_tokenizer/vocab.json",
    merges_filename="./santali_tokenizer/merges.txt"
)
example_text = "ᱵᱷᱟᱹᱭ ᱠᱩᱯᱩᱱᱟ" # Any Olchiki text

# Encode the text
output = tokenizer.encode(example_text)

# Print tokenized output
print("Tokens:", output.tokens)
print("Token IDs:", output.ids)

decoded_text = tokenizer.decode(output.ids)
print("Decoded Text:", decoded_text)
