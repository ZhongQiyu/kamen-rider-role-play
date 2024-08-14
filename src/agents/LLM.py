import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForSeq2SeqLM, BertTokenizer, BertForSequenceClassification, AutoModelForCausalLM
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu
import random
import json
import argparse

class UnifiedLLMHandler:
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
    
    def load_model(self, model_type="sequence_classification", num_labels=2):
        """Load a model based on the specified model type."""
        if model_type == "sequence_classification":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        elif model_type == "causal_lm":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        elif model_type == "seq2seq":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        else:
            raise ValueError("Unsupported model type.")
        print(f"Model {self.model_name} of type {model_type} loaded successfully.")

    def inference(self, input_text, max_length=50):
        """Perform inference using the loaded model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded first.")
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs['input_ids'], max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    class PEFTTrainer:
        def __init__(self, model_name, train_data, peft_method, num_labels=2, lr=5e-5, num_epochs=3):
            self.model_name = model_name
            self.train_data = train_data
            self.peft_method = peft_method
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            if torch.cuda.is_available():
                self.model.cuda()
            self.lr = lr
            self.num_epochs = num_epochs

            if peft_method == "freeze_layers":
                self.freeze_layers()
            elif peft_method == "adapter":
                self.add_adapter()

        def freeze_layers(self):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        def add_adapter(self):
            self.model.add_adapter("classification_adapter")
            self.model.train_adapter("classification_adapter")

        def preprocess_function(self, examples):
            return self.tokenizer(examples['texts'], truncation=True, padding=True)

        def train(self):
            dataset = HFDataset.from_dict(self.train_data)
            encoded_dataset = dataset.map(self.preprocess_function, batched=True)

            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=4,
                warmup_steps=10,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                learning_rate=self.lr,
            )

            def compute_metrics(p):
                preds = p.predictions.argmax(-1)
                return {'accuracy': (preds == p.label_ids).astype(float).mean().item()}

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=encoded_dataset,
                eval_dataset=encoded_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            self.save_model('./peft_model')

        def save_model(self, save_path):
            if self.peft_method == "adapter":
                self.model.save_adapter(save_path, "classification_adapter")
            else:
                self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
    
    class RAGHandler:
        def __init__(self, data):
            self.data = data
            self.dataset = HFDataset.from_dict(data)
            self.tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
            self.retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='exact', passages_path=None)
            self.model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq')

        def preprocess(self):
            return self.dataset.map(self._preprocess_function, batched=True)

        def _preprocess_function(self, examples):
            return self.tokenizer(examples['texts'], truncation=True, padding=True)

        def retrieve(self, query):
            input_ids = self.tokenizer(query, return_tensors="pt")["input_ids"]
            retrieved_docs = self.retriever(input_ids=input_ids, return_tensors="pt")
            retrieved_texts = [doc for doc in retrieved_docs['retrieved_texts'][0]]
            return retrieved_texts

        def generate_answer(self, query, retrieved_texts):
            context = " ".join(retrieved_texts)
            inputs = self.tokenizer.prepare_seq2seq_batch([query], context=[context], return_tensors='pt')
            outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], context_input_ids=inputs['context_input_ids'])
            answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return answer

    class RLHFTrainer:
        def __init__(self, data, reward_model, epochs=10, lr=0.001):
            self.data = data
            self.reward_model = reward_model
            self.epochs = epochs
            self.lr = lr

        def train_reward_model(self):
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.reward_model.parameters(), lr=self.lr)
            
            dataset = UnifiedLLMHandler.FeedbackDataset(self.data)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            for epoch in range(self.epochs):
                for batch in dataloader:
                    queries = [item['query'] for item in batch]
                    responses = [item['response'] for item in batch]
                    feedbacks = torch.tensor([item['feedback'] for item in batch], dtype=torch.float32).unsqueeze(1)
                    
                    query_embeddings = UnifiedLLMHandler.embed_text(queries)
                    response_embeddings = UnifiedLLMHandler.embed_text(responses)
                    
                    rewards = self.reward_model(response_embeddings)
                    
                    loss = criterion(rewards, feedbacks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")
        
    class FeedbackDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    class RewardModel(nn.Module):
        def __init__(self):
            super(UnifiedLLMHandler.RewardModel, self).__init__()
            self.fc = nn.Linear(768, 1)  # 假设嵌入维度为768
        
        def forward(self, x):
            return self.fc(x)
    
    @staticmethod
    def embed_text(texts):
        embeddings = torch.randn(len(texts), 768)
        return embeddings
    
    def train_rag(self, data):
        rag_handler = self.RAGHandler(data)
        query = "情報検索タスクに関する情報が必要です。"
        retrieved_texts = rag_handler.retrieve(query)
        print("Retrieved Texts:")
        for text in retrieved_texts:
            print(text)
        answer = rag_handler.generate_answer(query, retrieved_texts)
        print("Generated Answer:", answer)
    
    def train_rlhf(self, data):
        reward_model = self.RewardModel()
        rlhf_trainer = self.RLHFTrainer(data=data, reward_model=reward_model)
        rlhf_trainer.train_reward_model()
    
    def train_peft(self, model_name, train_data, peft_method):
        peft_trainer = self.PEFTTrainer(
            model_name=model_name,
            train_data=train_data,
            peft_method=peft_method
        )
        peft_trainer.train()

if __name__ == "__main__":
    # 示例用法
    handler = UnifiedLLMHandler(model_name='bert-base-uncased')
    
    # 加载并设置LLM基座
    handler.load_model(model_type="sequence_classification")
    
    # 进行推理
    response = handler.inference("Example text for inference.")
    print("Inference Response:", response)
    
    # Example data for RAG
    data = {
        "texts": [
            "これは情報検索タスクのための例文です。",
            "次の文を探しています。",
            "前の文と次の文の両方が必要です。",
            "これが私のリクエストです。",
            "情報検索は面白い分野です。"
        ]
    }
    
    handler.train_rag(data)
    
    # Example data for RLHF
    rlhf_data = [{'query': 'example', 'response': 'example', 'feedback': 1}]
    handler.train_rlhf(rlhf_data)
    
    # Example data for PEFT
    peft_data = {
        'texts': ['Example text'],
        'labels': [1]
    }
    handler.train_peft('bert-base-uncased', peft_data, peft_method='freeze_layers')
