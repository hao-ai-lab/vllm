from transformers import AutoTokenizer

# Load the tokenizer for DeepSeek R1
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

# Your input text
with open("playground/tokens.txt", "r") as f:
    input_text = f.read()

# Tokenize the input text
tokens = tokenizer.encode(input_text)

# Count the number of tokens
num_tokens = len(tokens)

print(f"Number of tokens: {num_tokens}")