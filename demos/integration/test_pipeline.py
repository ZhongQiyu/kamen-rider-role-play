# test_pipeline.py

from transformers import pipeline, BertForQuestionAnswering, BertTokenizer

class NLPTestPipeline:
    def __init__(self, text_classification_model_name, text_classification_tokenizer_name, qa_model_name):
        # Initialize models and tokenizers
        self.text_classification_pipeline = pipeline(
            "text-classification",
            model=text_classification_model_name,
            tokenizer=text_classification_tokenizer_name
        )
        
        self.qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)
        self.qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)
    
    def test_text_classification(self, text):
        # Perform text classification
        result = self.text_classification_pipeline(text)
        print("Text Classification Result:", result)
        return result
    
    def test_question_answering(self, question, context):
        # Perform question answering
        result = self.qa_pipeline(question=question, context=context)
        print("Question Answering Result:", result)
        return result

class TestRunner:
    def __init__(self):
        # Define model names and tokenizers
        self.text_classification_model_name = "model_name"
        self.text_classification_tokenizer_name = "tokenizer_name"
        self.qa_model_name = "hfl/chinese-roberta-wwm-ext"
        
        # Instantiate the NLPTestPipeline
        self.nlp_test_pipeline = NLPTestPipeline(
            text_classification_model_name=self.text_classification_model_name,
            text_classification_tokenizer_name=self.text_classification_tokenizer_name,
            qa_model_name=self.qa_model_name
        )
    
    def run_tests(self):
        # Test text classification
        print("Running text classification test:")
        self.nlp_test_pipeline.test_text_classification("这个手机真好用，我非常喜欢！")
        
        # Test question answering
        print("\nRunning question answering test:")
        self.nlp_test_pipeline.test_question_answering(
            question="这个手机好用吗？",
            context="这个手机真好用，我非常喜欢！"
        )

if __name__ == "__main__":
    test_runner = TestRunner()
    test_runner.run_tests()
