# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="elyza/ELYZA-japanese-Llama-2-7b")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b")
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b")
