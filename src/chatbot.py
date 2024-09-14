# chatbot.py

import ray
import json
import os
import random
import gradio as gr
import tensorflow as tf
import horovod.tensorflow as hvd
from ray import tune
from collections import defaultdict
from tensorflow.keras.models import Model
from ray.rllib.agents.ppo import PPOTrainer
from tensorflow.keras.layers import Input, Dense
from transformers import AutoModelForCausalLM, AutoTokenizerBertTokenizer, BertTokenizer, BertForQuestionAnswering, pipeline, AutoTokenizer

# 确保环境变量中存在OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("没有在环境变量中找到 OPENAI_API_KEY")

# 假定你的数据集和模型建构代码位于这里

# Placeholder for the model and dataset. Replace with actual code.
dataset = None  # Placeholder for the dataset. Replace with actual dataset's preparation code

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v1.0")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-13b-v1.0")

# Define the continuation generation function
def generate_continuation(prompt_text):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    generated_sequence = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_sequence

# Assume base_questions_and_answers is defined as shown earlier
# Prepare the dataset
def encode_qa_pairs(qa_pairs, tokenizer):
    inputs = defaultdict(list)
    for question, answer in qa_pairs:
        encoding = tokenizer(question, answer, return_tensors='tf', padding=True, truncation=True)
        for key, value in encoding.items():
            inputs[key].append(value[0])
    return {key: tf.convert_to_tensor(value) for key, value in inputs.items()}

# Use the encode_qa_pairs function to create the dataset
encoded_qa_pairs = encode_qa_pairs(base_questions_and_answers, tokenizer)

# A simple question answering model
def create_qa_model():
    # The input is the encoded question and answer pairs
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    token_type_ids = Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # BERT model
    bert_model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    sequence_output, pooled_output = bert_model(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )

    # Let's take the pooled_output for question answering
    qa_outputs = Dense(2, name="qa_outputs")(pooled_output)

    # This model will output the start and end logits for answer prediction
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=qa_outputs)

    return model

# Create the model
model = create_qa_model()

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
# 使用 `verbose=1` 在主工作进程打印详细信息
model.fit(dataset, epochs=3, callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)

# Initialize Horovod
hvd.init()

# 配置 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Add Horovod distributed callback
callbacks = [
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    # 如果有必要，可添加更多回调函数
]

# Horovod: 添加这一行来平均梯度

# Horovod: 仅在rank 0的worker上保存检查点和日志
your_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='path_to_save_model.h5')
your_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
if hvd.rank() == 0:
    callbacks.append(your_checkpoint_callback)
    callbacks.append(your_tensorboard_callback)

# Wrap the optimizer in Horovod's distributed optimizer
optimizer = tf.optimizers.Adam(0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

def create_labels_for_qa(context, answer, tokenizer):
    # 编码文本
    encodings = tokenizer.encode_plus(context, answer, return_offsets_mapping=True, truncation=True, padding="max_length")

    # 找到答案在编码后的文本中的起始和结束索引
    start_index = encodings.char_to_token(answer["start"])
    end_index = encodings.char_to_token(answer["end"] - 1)  # 因为结束索引是不包括在内的

    # 需要注意的是，如果答案不在文本中（因为可能截断），start_index和end_index可能是None
    # 这种情况下，我们可以将它们设置为0或编码长度减1
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(encodings["input_ids"]) - 1

    return start_index, end_index

# 假设你有一个函数来迭代你的数据集并应用上面的函数来创建标签
start_positions, end_positions = zip(*[create_labels_for_qa(context, answer, tokenizer) for context, answer in dataset])

# Assuming the targets are in the right shape and format for TensorFlow
dataset = tf.data.Dataset.from_tensor_slices((encoded_qa_pairs, [start_positions, end_positions]))
dataset = dataset.shuffle(len(base_questions_and_answers)).batch(8)

# Generate 1000 QA pairs
# Define a function to generate a random QA pair based on the base questions and answers
def generate_random_qa_pair(base_qa_list):
    # Randomly pick a QA pair from the base list
    return random.choice(base_qa_list)

# Generate and write QA pairs to a JSONL file
# Assuming base_questions_and_answers is a list of tuples with your base QA data
def write_qa_pairs_to_jsonl(qa_list, file_name):
    """Write question-answer pairs to a JSONL file."""
    with open(file_name, 'w', encoding='utf-8') as f:
        for qa in qa_list:
            # Generate a random QA pair
            question, answer = random.choice(qa_list)
            # Write the QA pair as a JSON object per line
            f.write(json.dumps({'question': question, 'answer': answer}) + '\n')

# 转换为字典格式，并确保一个问题可以有多个答案
qa_dict = defaultdict(list)
for question, answer in base_questions_and_answers:
    qa_dict[question].append(answer)

# 将生成的问答对写入到 JSONL 文件中
qa_pairs_file = 'qa_pairs.jsonl'  # Define your output JSONL file name
with open(qa_pairs_file, 'w', encoding='utf-8') as f:
    json.dump(qa_dict, f, ensure_ascii=False, indent=4)

# Call the function to write out 1000 QA pairs
write_qa_pairs_to_jsonl(base_questions_and_answers, qa_pairs_file)

print("QA pairs generation and writing to JSONL completed.")

# 准备数据集（用实际的数据集替换 None）
encoded_qa_pairs = None  # 这应当是经过编码后的问答对
print(f"1000 random QA pairs have been written to {qa_pairs_file}")

# 确保环境变量中存在OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("没有在环境变量中找到 OPENAI_API_KEY")

# 初始化 Horovod
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# 创建并编译 TensorFlow 模型
model.compile(optimizer=tf.optimizers.Adam(learning_rate=5e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# 创建问答 pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Horovod: 包装优化器
optimizer = hvd.DistributedOptimizer(tf.optimizers.Adam(0.001 * hvd.size()))

# 添加分布式训练的回调函数
callbacks = [
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# 仅在 worker 0 上保存检查点和日志
if hvd.rank() == 0:
    # 训练模型
    model.fit(encoded_qa_pairs, epochs=3, callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)

# Initialize Ray
ray.init()

# Run the Ray Tune training
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_workers": 4,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
    verbose=1 if hvd.rank() == 0 else 0
)

# 测试文本和问题
context = "《假面骑士剑》是一部很受欢迎的日本电视剧。"
test_questions = [
    "假面骑士剑的真名是什么？",
    "假面骑士剑的主要武器是什么？",
    # ... 更多测试问题
]

# 使用模型寻找答案
test_answers = qa_pipeline(question=test_questions, context=context)

# 输出结果
for question, answer in zip(test_questions, test_answers):
    print(f"问题: {question['question']}")
    print(f"答案: {answer['answer']}")

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_continuation,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type something..."),
    outputs="text",
    title="Japanese Text Continuation",
    description="Enter some text and the model will generate a continuation.",
    css=css
)

# Launch the interface
iface.launch()
