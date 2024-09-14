# test_speech_recognition.py

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import io

class SpeechRecognitionTest:
    def __init__(self, audio_file_path, language_code='en-US', sample_rate=16000):
        self.client = speech.SpeechClient()
        self.audio_file_path = audio_file_path
        self.language_code = language_code
        self.sample_rate = sample_rate

    def load_audio(self):
        with io.open(self.audio_file_path, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)
        return audio

    def configure(self):
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code
        )
        return config

    def recognize_speech(self):
        audio = self.load_audio()
        config = self.configure()
        response = self.client.recognize(config=config, audio=audio)
        return response

    def print_transcripts(self, response):
        for result in response.results:
            print('Transcript: {}'.format(result.alternatives[0].transcript))

    def test_recognition(self):
        response = self.recognize_speech()
        self.print_transcripts(response)

if __name__ == "__main__":
    # Example usage
    tester = SpeechRecognitionTest(audio_file_path='audio_file.wav')
    tester.test_recognition()
