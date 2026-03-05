# LOGGING NOW COMPLETED IN GOOGLE COLAB NOTEBOOK (colab_1_run.ipynb) INSTEAD OF THIS .PY FILE
# Login to Hugging Face Hub using your token (set as environment variable)
# import os
# from huggingface_hub import login
# # Set your Hugging Face token as an environment variable: $env:HF_TOKEN="your_token_here"
# login(token=os.getenv("HF_TOKEN"))

from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "meta-llama/Llama-3.2-3B-Instruct" # Change to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Tokenizer and model loaded successfully!")

# Check tokenizer vocab size
print("Vocabulary size of the tokenizer: ", tokenizer.vocab_size)
# Token decoding test
token_id = 100000 # Example token ID (adjust based on your tokenizer's vocab size)
print(f"Token ID {token_id} corresponds to token: '{tokenizer.decode([token_id])}'")
# Decode multiple token IDs
token_ids = [100000, 100001, 100002] # Example token IDs
print(f"Token IDs {token_ids} correspond to tokens: '{tokenizer.decode(token_ids)}'")

#Token encoding test
Text = "Hello, how are you?"
tokens = tokenizer.encode(Text, add_special_tokens=False)
print(f"Text: '{Text}' is encoded to token IDs: {tokens}")


