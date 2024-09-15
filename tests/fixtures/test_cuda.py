# test_cuda.py

import torch
import fugashi

class TestCUDA:
    def __init__(self):
        self.tokenizer = fugashi.Tagger()

    def test_cuda_availability(self):
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Device:", torch.cuda.get_device_name(device))
        else:
            print("CUDA is not available.")

    def test_fugashi_tokenizer(self, text):
        # 测试Fugashi分词器
        tokens = [token.surface for token in self.tokenizer(text)]
        print("Tokenized text:", tokens)
        return tokens

    def run_all_tests(self):
        self.test_cuda_availability()
        sample_text = "これは日本語のテキストです。"
        self.test_fugashi_tokenizer(sample_text)


if __name__ == "__main__":
    tester = CUDAFugashiTester()
    tester.run_all_tests()
