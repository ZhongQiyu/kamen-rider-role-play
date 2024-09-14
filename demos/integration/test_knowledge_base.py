# test_knowledge_base.py

import os
import ssl
import json
import unittest
import nltk
import numpy as np
from faiss_indexer import FaissIndexer
from knowledge_base import KnowledgeBase

# 解决nltk下载证书验证失败问题
ssl._create_default_https_context = ssl._create_unverified_context

class KnowledgeBaseLoader:
    def __init__(self, file_path='dialogues.json'):
        self.file_path = file_path
        self.dialogues = self.load_from_file()

    @staticmethod
    def setup_nlp_resources():
        nltk.download('punkt')

    def load_from_file(self):
        if not os.path.exists(self.file_path):
            return {}
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup NLP resources
        print("Setting up NLP resources...")
        KnowledgeBaseLoader.setup_nlp_resources()

        # Initialize knowledge base and Faiss indexer
        print("Initializing knowledge base and Faiss indexer...")
        cls.kb_loader = KnowledgeBaseLoader()
        cls.indexer = FaissIndexer(dimension=768)
        print("Knowledge base and Faiss indexer initialized successfully.")

    def test_vectors_storage_and_loading(self):
        # Store and load example vectors
        vectors = np.random.rand(10, 768).astype('float32')
        ids = [f'vector_{i}' for i in range(10)]
        self.indexer.add_vectors(vectors, ids)

        loaded_vectors = self.indexer.load_vectors_from_redis(ids)
        print("Vectors loaded from Redis.")

        # Assert that the loaded vectors are not None
        self.assertIsNotNone(loaded_vectors)

    def test_faiss_index_creation_and_query(self):
        # Create and query Faiss index
        query_vector = np.random.rand(1, 768).astype('float32')
        D, I = self.indexer.search(query_vector)
        print(f"Query vector: {query_vector}")
        print(f"Distances: {D}")
        print(f"Indices: {I}")

        # Assert that the indices and distances are valid
        self.assertIsNotNone(I)
        self.assertIsNotNone(D)
        self.assertGreater(len(I), 0)
        self.assertGreater(len(D), 0)

    def test_knowledge_base_loading(self):
        # Test loading of the knowledge base
        dialogues = self.kb_loader.dialogues
        print(f"Loaded dialogues: {dialogues}")

        # Assert that dialogues are loaded correctly
        if os.path.exists(self.kb_loader.file_path):
            self.assertIsInstance(dialogues, dict)
        else:
            self.assertEqual(dialogues, {})

if __name__ == "__main__":
    unittest.main()
