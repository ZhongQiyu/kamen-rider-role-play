# bert_qa.py

import json
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline

def generate_qa_pairs():
    """生成 1000 个问答对并保存到 JSON 文件。"""
    qa_pairs = [{"question": "什么是Python?", "answer": "Python是一种编程语言。"} for _ in range(1000)]
    with open('qa_pairs.jsonl', 'w') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + '\n')

def setup_bert_model():
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp