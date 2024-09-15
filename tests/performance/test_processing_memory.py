# test_processing_memory.py

import torch
import torch.nn as nn
import torch.distributed as dist
import psutil
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TestProcessingMemory:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = CustomT5Model.from_pretrained(model_name)
    
    def test_model_processing(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output = self.model(input_ids)
        return output
    
    def monitor_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
        }

    def setup_ddp(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup_ddp(self):
        dist.destroy_process_group()

    def ddp_training_step(self, rank, world_size, input_text):
        self.setup_ddp(rank, world_size)
        model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output = model(input_ids)
        
        # Custom all-reduce operation on the output (for example purposes)
        dist.all_reduce(output.logits, op=dist.ReduceOp.SUM)
        
        memory_usage = self.monitor_memory_usage()
        print(f"Memory Usage: {memory_usage}")
        
        self.cleanup_ddp()

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

# Example usage:
if __name__ == "__main__":
    tester = TestProcessingMemory()
    tester.test_model_processing("This is a test sentence.")

    # Example multi-process execution
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(tester.ddp_training_step, args=(world_size, "This is a test sentence."), nprocs=world_size, join=True)

   