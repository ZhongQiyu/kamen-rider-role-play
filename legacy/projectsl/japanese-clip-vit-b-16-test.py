# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="rinna/japanese-clip-vit-b-16")

# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("rinna/japanese-clip-vit-b-16")
model = AutoModelForZeroShotImageClassification.from_pretrained("rinna/japanese-clip-vit-b-16")
