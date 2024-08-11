# aligner.py

import re
import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, RagTokenizer, RagRetriever, RagSequenceForGeneration, T5Tokenizer, T5Model
from sentence_transformers import SentenceTransformer
from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer
from typing import List
import language_tool_python

class JapaneseTextProcessor:
    def __init__(self):
        # 初始化中文语法检查工具
        self.language_tool = language_tool_python.LanguageTool('zh-CN')
        # 初始化日语处理器
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForTokenClassification.from_pretrained('cl-tohoku/bert-base-japanese')
        self.fugashi_tagger = FugashiTagger()
        self.janome_tokenizer = JanomeTokenizer()
        # 初始化对话生成器
        self.retriever_model_name = "facebook/dpr-ctx_encoder-multiset-base"
        self.rag_model_name = "facebook/rag-sequence-nq"
        self.retriever = SentenceTransformer(self.retriever_model_name)
        self.rag_tokenizer = RagTokenizer.from_pretrained(self.rag_model_name)
        self.rag_retriever = RagRetriever.from_pretrained(self.rag_model_name, index_name="exact", use_dummy_dataset=True)
        self.rag_model = RagSequenceForGeneration.from_pretrained(self.rag_model_name)
        # 初始化T5 Embeddings
        self.t5_tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-t5-small")
        self.t5_model = T5Model.from_pretrained("rinna/japanese-t5-small")

    def correct_chinese(self, text):
        matches = self.language_tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def bert_tokenize(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return tokens, predictions[0].tolist()

    def fugashi_tokenize(self, text):
        return [word.surface for word in self.fugashi_tagger(text)]

    def janome_tokenize(self, text):
        return [token.surface for token in self.janome_tokenizer.tokenize(text)]

    def align_grammar(self, text):
        processed_text = self.preprocess_text(text)
        bert_tokens, bert_predictions = self.bert_tokenize(processed_text)
        fugashi_tokens = self.fugashi_tokenize(processed_text)
        janome_tokens = self.janome_tokenize(processed_text)
        alignment = {
            "original_text": text,
            "processed_text": processed_text,
            "bert_tokens": bert_tokens,
            "bert_predictions": bert_predictions,
            "fugashi_tokens": fugashi_tokens,
            "janome_tokens": janome_tokens
        }
        return alignment

    def generate_response(self, question, context_documents):
        inputs = self.rag_tokenizer(question, return_tensors="pt")
        question_embeddings = self.retriever.encode([question], convert_to_tensor=True)
        docs = self.rag_retriever(question_inputs=inputs['input_ids'], prefix_allowed_tokens_fn=None)
        outputs = self.rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=docs['context_input_ids'])
        response = self.rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def generate_embeddings(self, text):
        inputs = self.t5_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.t5_model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings

    def align_and_embed(self, text):
        alignment = self.align_grammar(text)
        embeddings = self.generate_embeddings(text)
        alignment["embeddings"] = embeddings.tolist()
        return alignment

if __name__ == "__main__":
    processor = JapaneseTextProcessor()
    text = "这是一个用于测试的文本。"
    corrected_text = processor.correct_chinese(text)
    print("Corrected Text:", corrected_text)
    # 其他方法可以按照实际需求调用
