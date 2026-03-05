import os
from huggingface_hub import login
# Set your Hugging Face token as an environment variable: $env:HF_TOKEN="your_token_here"
login(token=os.getenv("HF_TOKEN"))

model_id = "meta-llama/llama-3.2-3B_Instruct"