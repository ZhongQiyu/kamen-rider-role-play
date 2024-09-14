# test_loading_performance.py

import time
import torch
from transformers import T5ForConditionalGeneration

class TestLoadingPerformance:
    def __init__(self, model_name: str = "t5-small"):
        self.model_name = model_name
        self.model = None
        self.load_time = None

    def load_model(self):
        start_time = time.time()
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.load_time = time.time() - start_time
        print(f"Model {self.model_name} loaded in {self.load_time:.4f} seconds.")

    def test_inference_speed(self, input_ids):
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.4f} seconds.")
        return outputs

    def test_loading_performance(self):
        self.load_model()
        # Here, you can add additional performance tests
        # For example, you can check memory usage, GPU utilization, etc.
        print("Additional performance metrics can be added here.")

# Example usage:
if __name__ == "__main__":
    tester = TestLoadingPerformance(model_name="t5-small")
    tester.test_loading_performance()
    
    # Example input for testing inference speed
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    input_text = "This is a test sentence."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    tester.test_inference_speed(input_ids)
