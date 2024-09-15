# test_openai_api_key.py

import unittest
import os
import openai

class TestOpenAIAPI(unittest.TestCase):
    def setUp(self):
        # Setup code to run before each test
        self.api_key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def test_openai_api_key_exists(self):
        # Test if API key is correctly retrieved
        self.assertIsNotNone(self.api_key, "API key should not be None")

    def test_openai_completion(self):
        # Test the OpenAI API completion call
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Hello, world!",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        self.assertIsNotNone(response, "Response should not be None")
        self.assertIn('choices', response, "Response should contain 'choices'")
        self.assertTrue(len(response['choices']) > 0, "Response should have at least one choice")

if __name__ == '__main__':
    unittest.main()
