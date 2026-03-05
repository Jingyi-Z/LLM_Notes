# LOGGING NOW COMPLETED IN GOOGLE COLAB NOTEBOOK (colab_1_run.ipynb) INSTEAD OF THIS .PY FILE
# Login to Hugging Face Hub using your token (set as environment variable)
# import os
# from huggingface_hub import login
# # Set your Hugging Face token as an environment variable: $env:HF_TOKEN="your_token_here"
# login(token=os.getenv("HF_TOKEN"))

# MODEL LOADING AND TOKENIZER SETTING
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-3B-Instruct" # Change to your desired model
cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/content/drive/MyDrive/hf_cache/transformers")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
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

#__________________________________________________________________________
# Autoregression task: Generate text based on a prompt
import torch
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt") # "pt" stands for PyTorch tensor format
print("Model input IDs：",input_ids)

# Inspect the possible outputs of the model
outputs = model(input_ids)
# outputs.logits is the confidence score for each token at each position (not yet converted to probabilities)
# outputs.logits has a shape of (batch_size, sequence_length, vocab_size), where:
# - batch_size is the number of input sequences (in this case, 1)
# - sequence_length is the length of the input sequence (number of tokens in the prompt)
print("Model outputs (logits) shape:", outputs.logits.shape) # Should be (batch_size, sequence_length, vocab_size)


# Inspect the logits for the last token in the input sequence
last_logits = outputs.logits[:, -1, :] 
probabilities = torch.softmax(last_logits, dim=-1) # Convert logits to probabilities using softmax

# Get the top 5 predicted token IDs and their probabilities
top_k = 5
top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
print(f"Top {top_k} predicted tokens:")

for i in range(top_k):
    token_id = top_k_indices[0, i].item() # Get the token ID
    token_prob = top_k_probs[0, i].item() # Get the probability of the token
    token_str = tokenizer.decode([token_id]) # Decode the token ID to a string
    print(f"Token ID: {token_id}, Token: '{token_str}', Probability: {token_prob:.4f}")

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: '{prompt}'\nGenerated Text: '{generated_text}'")