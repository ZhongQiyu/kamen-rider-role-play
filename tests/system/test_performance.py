# test_performance.py

import torch
import torch.nn as nn
import time
from transformers import T5Tokenizer
from custom_t5_model import CustomT5Model  # Assuming the custom model is in this module

class T5PerformanceTester:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = CustomT5Model.from_pretrained(model_path).to(self.device)
    
    def prepare_inputs(self, sentences):
        return self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
    
    def measure_inference_time(self, inputs):
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_ids=inputs.input_ids)
            end_time = time.time()
            inference_time = end_time - start_time
        return inference_time

    def measure_memory_usage(self, inputs):
        self.model.eval()
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats(self.device)
            outputs = self.model(input_ids=inputs.input_ids)
            memory_usage = torch.cuda.max_memory_allocated(self.device)
        return memory_usage

    def measure_accuracy(self, inputs, expected_outputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == expected_outputs).float().mean().item()
        return accuracy

    def run_tests(self, test_sentences, expected_outputs=None):
        inputs = self.prepare_inputs(test_sentences)
        
        print("Running performance tests...\n")
        
        # Measure inference time
        inference_time = self.measure_inference_time(inputs)
        print(f"Inference Time: {inference_time:.4f} seconds")
        
        # Measure memory usage
        memory_usage = self.measure_memory_usage(inputs)
        print(f"Memory Usage: {memory_usage / (1024 ** 2):.2f} MB")
        
        # (Optional) Measure accuracy if expected outputs are provided
        if expected_outputs is not None:
            accuracy = self.measure_accuracy(inputs, expected_outputs)
            print(f"Accuracy: {accuracy * 100:.2f}%")
    
    @staticmethod
    def from_args():
        import argparse
        parser = argparse.ArgumentParser(description="Test model performance")
        parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
        parser.add_argument("--input_text", type=str, nargs='+', required=True, help="Input text for testing")
        parser.add_argument("--expected_outputs", type=int, nargs='*', help="Expected outputs for accuracy measurement")
        args = parser.parse_args()
        return T5PerformanceTester(args.model_path), args.input_text, args.expected_outputs

if __name__ == "__main__":
    tester, test_sentences, expected_outputs = T5PerformanceTester.from_args()
    tester.run_tests(test_sentences, expected_outputs)
