# chatbot.py

import os
import time
import logging
import hmac
import hashlib
import base64
import torch
import spacy
import gradio as gr
import pysrt
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from ray import tune
from ray.tune import JupyterNotebookReporter
from emotion_recognition import EmotionRecognizer  # 确保这个模块是可用的

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KamenRiderChatbot:
    def __init__(self, appid, secret_key, model_name='sonoisa/t5-base-japanese', data_dir=None, model_dir='./t5-finetuned'):
        self.appid = appid
        self.secret_key = secret_key
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # 加载或初始化 T5 模型和分词器
        if os.path.exists(model_dir):
            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_dir)
        else:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

        # 初始化 Spacy 和 Emotion Recognizer
        self.nlp = spacy.load('en_core_web_sm')
        self.emotion_recognizer = EmotionRecognizer()
        
        # 初始化日志和 Gradio
        self.chat_history = []

        # 如果提供了数据目录，则进行微调
        if data_dir:
            self.fine_tune_model()

    def generate_signature(self):
        """生成 API 请求签名"""
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((self.appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        signa = hmac.new(self.secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa).decode('utf-8')
        return signa, ts

    def generate_text(self, prompt, max_length=50):
        """使用 T5 生成文本"""
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors='pt')
        output = self.t5_model.generate(input_ids, max_length=max_length)
        return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)

    def format_chat_prompt(self, message):
        """格式化对话历史为 Prompt"""
        prompt = ""
        for turn in self.chat_history:
            user_message, bot_message = turn
            prompt = f"{prompt}\nユーザー: {user_message}\nアシスタント: {bot_message}"
        prompt = f"{prompt}\nユーザー: {message}\nアシスタント:"
        return prompt

    def analyze_emotion(self, input_text):
        """情感分析"""
        emotion = self.emotion_recognizer.recognize(input_text)
        return emotion

    def send_message(self, message):
        """处理用户消息并生成响应"""
        logging.info(f"受信メッセージ: {message}")

        formatted_prompt = self.format_chat_prompt(message)
        bot_message = self.generate_text(formatted_prompt)
        self.chat_history.append((message, bot_message))
        
        return bot_message

    def extract_lines_from_srt(self, srt_file):
        """从SRT文件中提取文本行"""
        subs = pysrt.open(srt_file)
        lines = []
        for sub in subs:
            lines.append(sub.text.strip())
        return lines

    def generate_emotional_response(self, input_text):
        """根据情感生成响应"""
        emotion = self.analyze_emotion(input_text)
        if emotion == "anger":
            return "私はあなたが少しイライラしていることに気付きました。何かお手伝いできますか？"
        elif emotion == "happiness":
            return "あなたが満足しているのがとても嬉しいです！もっとお手伝いできることがありますか？"
        else:
            return self.send_message(input_text)

    def respond(self, message):
        """响应用户输入"""
        response = self.generate_emotional_response(message)
        return "", response

    def preprocess_data(self):
        """数据预处理，用于微调"""
        input_texts = []
        target_texts = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                for line in lines:
                    text = line.strip()
                    if text:  # 确保文本不为空
                        input_texts.append(text)
                        target_texts.append(text)

        dataset = Dataset.from_dict({"input_text": input_texts, "target_text": target_texts})
        return dataset.train_test_split(test_size=0.1)

    def preprocess_function(self, examples):
        """为模型输入预处理"""
        inputs = self.t5_tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
        labels = self.t5_tokenizer(text_target=examples['target_text'], max_length=512, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    def fine_tune_model(self, num_train_epochs=3, per_device_train_batch_size=4, learning_rate=5e-5):
        """微调模型"""
        datasets = self.preprocess_data()
        if datasets:
            train_dataset = datasets['train']
            eval_dataset = datasets['test']

            train_dataset = train_dataset.map(self.preprocess_function, batched=True)
            eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)

            training_args = TrainingArguments(
                output_dir=self.model_dir,
                evaluation_strategy="steps",
                learning_rate=learning_rate,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                num_train_epochs=num_train_epochs,
                weight_decay=0.01,
                save_steps=10_000,
                save_total_limit=2,
                logging_dir='./logs',
                logging_steps=200,
                fp16=torch.cuda.is_available(),
                eval_steps=500,
            )

            trainer = Trainer(
                model=self.t5_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            start_time = time.time()  # 开始计时
            trainer.train()  # 开始训练
            end_time = time.time()  # 结束计时

            elapsed_time = end_time - start_time
            total_steps = len(train_dataset) // per_device_train_batch_size * num_train_epochs

            print(f"Total training time: {elapsed_time:.2f} seconds for {total_steps} steps")
            print(f"Average time per step: {elapsed_time / total_steps:.2f} seconds/step")

            # 保存微调后的模型
            self.save_model()

    def hyperparameter_search(self, train_dataset, eval_dataset, search_space, num_samples=10):
        """超参数搜索"""
        def train_tune(config):
            trainer = self.train(
                train_dataset,
                eval_dataset,
                num_train_epochs=config['num_train_epochs'],
                per_device_train_batch_size=config['per_device_train_batch_size'],
                learning_rate=config['learning_rate']
            )
            eval_results = trainer.evaluate(eval_dataset)
            tune.report(mean_accuracy=eval_results.get("eval_accuracy", 0.0))

        analysis = tune.run(
            train_tune,
            resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
            config=search_space,
            metric="mean_accuracy",
            mode="max",
            num_samples=num_samples,
            progress_reporter=JupyterNotebookReporter()
        )

        if analysis:
            best_trial = analysis.best_trial
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final accuracy: {best_trial.last_result.get('mean_accuracy', 0.0)}")
            return best_trial

    def save_model(self):
        """保存微调后的模型"""
        self.t5_model.save_pretrained(self.model_dir)
        self.t5_tokenizer.save_pretrained(self.model_dir)

    def launch_gradio(self):
        """启动 Gradio 界面"""
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(height=240) 
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

            btn.click(self.respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            msg.submit(self.respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

            gr.Markdown("""<h1><center>Kamen Rider Blade Roleplay Chatbot</center></h1>""")

        demo.launch(share=True)

if __name__ == '__main__':
    appid = "your_appid_here"
    secret_key = "your_secret_key_here"
    data_dir = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\text\\txt\\'  # 可选的数据目录

    chatbot = KamenRiderChatbot(appid, secret_key, data_dir=data_dir)
    chatbot.launch_gradio()
