# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="stabilityai/japanese-stable-clip-vit-l-16", trust_remote_code=True)

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("stabilityai/japanese-stable-clip-vit-l-16", trust_remote_code=True)
