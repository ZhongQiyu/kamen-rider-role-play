# test_multi_modal.py

import torch
from heron.models.video_blip import VideoBlipForConditionalGeneration, VideoBlipProcessor
from transformers import LlamaTokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel
from PIL import Image
import requests

# Set device
device_id = 0
device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

# Load VideoBLIP model and processor
MODEL_NAME = "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1"
model = VideoBlipForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, ignore_mismatched_sizes=True
)
model = model.half().eval().to(device)

processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
processor.tokenizer = tokenizer

# Prepare image input
url = "https://www.barnorama.com/wp-content/uploads/2016/12/03-Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = f"##human: この画像の面白い点は何ですか?\n##gpt: "

# Preprocess input
inputs = processor(
    text=text,
    images=image,
    return_tensors="pt",
    truncation=True,
)
inputs = {k: v.to(device) for k, v in inputs.items()}
inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float16)

# Set end of sequence tokens
eos_token_id_list = [
    processor.tokenizer.pad_token_id,
    processor.tokenizer.eos_token_id,
    int(tokenizer.convert_tokens_to_ids("##"))
]

# Perform inference with VideoBLIP model
with torch.no_grad():
    out = model.generate(
        **inputs, 
        max_length=256, 
        do_sample=False, 
        temperature=0.0, 
        eos_token_id=eos_token_id_list, 
        no_repeat_ngram_size=2
    )

print("VideoBLIP Output:", processor.tokenizer.batch_decode(out))

# Text generation using ELYZA Japanese LLaMA-2-7b
text_gen_pipe = pipeline("text-generation", model="elyza/ELYZA-japanese-Llama-2-7b")

tokenizer_elyza = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b")
model_elyza = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b")

# Example text generation
generated_text = text_gen_pipe("こんにちは、AIの未来についてどう思いますか？", max_length=50)
print("ELYZA Text Generation Output:", generated_text)

# Feature extraction using Japanese CLIP model (rinna/japanese-clip-vit-b-16)
feature_extraction_pipe = pipeline("feature-extraction", model="rinna/japanese-clip-vit-b-16")

processor_clip = AutoProcessor.from_pretrained("rinna/japanese-clip-vit-b-16")
model_clip = AutoModelForZeroShotImageClassification.from_pretrained("rinna/japanese-clip-vit-b-16")

# Example image feature extraction
image_features = feature_extraction_pipe(image)
print("Feature Extraction Output (rinna CLIP):", image_features)

# Feature extraction using Stability AI's Japanese Stable CLIP
stable_clip_pipe = pipeline("feature-extraction", model="stabilityai/japanese-stable-clip-vit-l-16", trust_remote_code=True)

model_stable_clip = AutoModel.from_pretrained("stabilityai/japanese-stable-clip-vit-l-16", trust_remote_code=True)

# Example usage with Stability AI's Japanese Stable CLIP
stable_clip_features = stable_clip_pipe(image)
print("Feature Extraction Output (Stability AI CLIP):", stable_clip_features)
