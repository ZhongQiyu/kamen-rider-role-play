# t5_demo.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5模型和分词器
model_name = "t5-small"  # 替换为实际的模型名称
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 如果有GPU可用，则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义文本生成函数
def generate_text(input_text, max_length=50):
    # 将输入文本转换为张量
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # 使用模型生成文本
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    # 解码生成的文本
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# 测试生成
input_text = "Translate the following English text to Japanese: Hello, how are you?"
output = generate_text(input_text)
print("Generated Output:", output)

# Step 1: Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 2: Load the Shusheng Puyu 7B model and tokenizer
model_name = "shusheng-puyu-7b"  # Replace with actual model path or identifier

# Use the bf16 data type if available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Load the tokenizer and model with appropriate precision
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True  # Use this flag for reducing memory usage
).to(device)

# Step 3: Define a function to perform quantized inference
def generate_response(input_text, max_length=100):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate response using model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,  # Enable sampling for varied responses
            top_p=0.9,  # Top-p sampling
            temperature=0.7  # Temperature setting to control randomness
        )
    
    # Decode and return the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 4: Test the inference with a sample input
if __name__ == "__main__":
    sample_input = "你好，请问你有什么推荐的书籍吗？"
    response = generate_response(sample_input)
    print(f"Model response: {response}")

# custom_ft_t5.py

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import T5ForConditionalGeneration, T5Tokenizer

class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config):
        super(CustomT5Model, self).__init__(config)
        
        # Custom word and position embeddings
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Create word embeddings
        word_embeddings = self.word_embedding(input_ids)
        
        # Create position embeddings
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine word and position embeddings
        embeddings = word_embeddings + position_embeddings.unsqueeze(0)
        
        # Custom mask operation (example: modify attention mask)
        if attention_mask is not None:
            attention_mask = self.custom_mask_operation(attention_mask)

        # Call the parent class's forward method with modified embeddings and mask
        return super().forward(inputs_embeds=embeddings, attention_mask=attention_mask, **kwargs)
    
    def custom_mask_operation(self, attention_mask):
        # Define your custom mask operation here
        custom_mask = attention_mask.clone()
        # Custom mask logic here...
        return custom_mask


class DistributedTrainingManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.world_size = torch.cuda.device_count()

    def setup_ddp(self, rank):
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)

    def cleanup_ddp(self):
        dist.destroy_process_group()

    def ddp_training_step(self, rank, input_text):
        self.setup_ddp(rank)

        # Prepare input data
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        # Wrap the model in DistributedDataParallel
        model_ddp = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])

        # Forward pass
        output = model_ddp(input_ids)

        # Custom all-reduce operation on the output
        dist.all_reduce(output.logits, op=dist.ReduceOp.SUM)

        self.cleanup_ddp()

    def run(self, input_text):
        torch.multiprocessing.spawn(
            self.ddp_training_step, args=(input_text,), nprocs=self.world_size, join=True
        )


# Initialize the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = CustomT5Model.from_pretrained("t5-small")

# Initialize the distributed training manager
manager = DistributedTrainingManager(model, tokenizer)

# Example input text
input_text = "This is a test sentence."

# Run the distributed training
manager.run(input_text)

# test_t5.py

from transformers import T5ForConditionalGeneration, T5Tokenizer, BertJapaneseTokenizer, BertModel
import torch

# 初始化T5模型和tokenizer
t5_model_name = "sonoisa/t5-base-japanese"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# 初始化BERT模型和tokenizer
bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

def generate_similar_sentence(input_text, num_return_sequences=3, max_length=50):
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
    
    outputs = t5_model.generate(input_ids, 
                                max_length=max_length, 
                                num_return_sequences=num_return_sequences, 
                                num_beams=5, 
                                no_repeat_ngram_size=2, 
                                early_stopping=True)
    
    generated_texts = []
    for output in outputs:
        text = t5_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if text.strip() and text != input_text:  # 过滤空白文本或重复生成的文本
            generated_texts.append(text)
    
    return generated_texts

def get_sentence_embedding(sentence):
    inputs = bert_tokenizer(sentence, return_tensors='pt')
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def find_most_similar_sentence(original_sentence, candidate_sentences):
    original_embedding = get_sentence_embedding(original_sentence)
    best_match = None
    max_similarity = -float('inf')

    for sentence in candidate_sentences:
        candidate_embedding = get_sentence_embedding(sentence)
        similarity = torch.cosine_similarity(original_embedding, candidate_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = sentence

    return best_match

# 示例对话
original_sentence = "君は本当に馬鹿だな。"

# 生成相似句子
similar_sentences = generate_similar_sentence(original_sentence)

# 输出生成的相似句子
print("生成的相似句子:")
for idx, sentence in enumerate(similar_sentences):
    print(f"Similar Sentence {idx+1}: {sentence}")

# 上下文信息和角色背景信息
context_template = """
【場面説明】：
- 夜の森の中、月明かりが木々の隙間から差し込む。彼らは敵の影に気づいているが、行動を起こすべきか迷っている。

【キャラクター】:
- 橘 朔也：冷静かつ正義感が強い、だが過去に裏切りの経験があり、常に葛藤している。

【対話】:
- 橘 朔也：「彼が変身したら、世界はどうなるんだろう？」
"""

character_background_template = """
【橘 朔也】:
- 年齢：30歳
- 性格：冷静、正義感が強いが、内心には深い苦悩を抱えている。
- 役割：物語の中での指導者的存在、仲間たちを守ることに全力を尽くす。
- 過去の経験：以前の戦いで仲間を失い、それ以降信頼することを恐れるようになった。
- 典型的なセリフ：「これは俺の使命だ、誰にも邪魔させない。」
"""

# 假设有一组相似的句子库
candidate_sentences = [
    "彼が変われば、世界も変わるかもしれない。",
    "彼が変わると、何が起こるのか。",
    "彼が変身することで、世界が変わるかも。",
    "彼が変わるなら、全てが違って見えるだろう。"
]

# 找出与原句子最相似的句子
best_match = find_most_similar_sentence(original_sentence, candidate_sentences)

print(f"\n与原句子最相似的句子: {best_match}")

# 输出上下文信息和角色背景信息
print("\n上下文信息:")
print(context_template)

print("\n角色背景信息:")
print(character_background_template)
