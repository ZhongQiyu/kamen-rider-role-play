# asr.py

import re
import os
import json
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import wiener
import torch
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
from flask import Flask, request, jsonify
from threading import Thread
import time

class ASR:
    def __init__(self, model_name="cl-tohoku/bert-base-japanese", db_path=None):
        self.app = Flask(__name__)
        self.model_name = model_name
        self.messages = {}
        self.processed_data_store = {}
        self.tokenizer, self.model = self.load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = []
        self.file_path = None
        self.audio = None
        self.sr = None
        self.processed_audio = None

        if db_path:
            self.file_path = db_path
            if self.is_audio_file(db_path):
                self.audio, self.sr = sf.read(db_path, dtype='float32')
                self.processed_audio = self.audio.copy()

    @staticmethod
    def is_audio_file(file_path):
        return any([file_path.endswith(fmt) for fmt in ['.mp3', '.m4a', '.wav']])

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = BertForMaskedLM.from_pretrained(self.model_name)
        return tokenizer, model

    def async_communication(self, agent_id, message, callback_url):
        time.sleep(2)  # Simulate delay
        if agent_id not in self.messages:
            self.messages[agent_id] = []
        self.messages[agent_id].append({'from': 'system', 'message': message})

    def async_data_processing(self, data_id, raw_data, callback_url):
        time.sleep(5)  # Simulate processing delay
        self.processed_data_store[data_id] = {'processed_data': raw_data}

    def setup_routes(self):
        @self.app.route('/agent_comm', methods=['POST'])
        def agent_comm():
            data = request.json
            agent_id = data['agent_id']
            message = data['message']
            callback_url = data.get('callback_url')

            if callback_url:
                thread = Thread(target=self.async_communication, args=(agent_id, message, callback_url))
                thread.start()
                return jsonify({'confirmation_message': 'Asynchronous communication initiated.'})
            else:
                if agent_id not in self.messages:
                    self.messages[agent_id] = []
                self.messages[agent_id].append({'from': 'system', 'message': message})
                return jsonify({'confirmation_message': 'Communication initiated.'})

        @self.app.route('/agent_comm', methods=['GET'])
        def get_agent_messages():
            agent_id = request.args.get('agent_id')
            return jsonify({'messages': self.messages.get(agent_id, [])})

        @self.app.route('/data_processor', methods=['POST'])
        def data_processor():
            data = request.json
            raw_data = data['raw_data']
            data_id = str(len(self.processed_data_store) + 1)
            callback_url = data.get('callback_url')

            if callback_url:
                thread = Thread(target=self.async_data_processing, args=(data_id, raw_data, callback_url))
                thread.start()
                return jsonify({'processing_id': data_id})
            else:
                self.processed_data_store[data_id] = {'processed_data': raw_data}
                return jsonify({'processing_id': data_id})

        @self.app.route('/data_processor/status', methods=['GET'])
        def data_processor_status():
            processing_id = request.args.get('processing_id')
            status = 'completed' if processing_id in self.processed_data_store else 'processing'
            return jsonify({'status': status})

        @self.app.route('/data_processor/result', methods=['GET'])
        def data_processor_result():
            processing_id = request.args.get('processing_id')
            return jsonify(self.processed_data_store.get(processing_id, {}))

    def run_flask(self):
        self.setup_routes()
        self.app.run(debug=True)

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data.append(file.read())

    def export_to_txt(self, output_txt):
        with open(output_txt, 'w', encoding='utf-8') as file:
            for content in self.data:
                file.write(content + '\n')

    def process_all_files(self, directory_path):
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        files = sorted(files, key=self.sort_files)
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            self.process_file(file_path)

    @staticmethod
    def sort_files(filename):
        part = filename.split('.')[0]
        try:
            return int(part)
        except ValueError:
            return float('inf')

    def calculate_snr(self):
        noise_part = self.audio[0:int(0.5 * self.sr)]
        signal_power = np.mean(self.audio ** 2)
        noise_power = np.mean(noise_part ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def apply_noise_reduction_if_needed(self, threshold_snr=10):
        snr = self.calculate_snr()
        if snr < threshold_snr:
            print(f"SNRが{snr} dBのため、ノイズリダクションを適用します")
            self.audio = wiener(self.audio)
        else:
            print(f"SNRが{snr} dBのため、ノイズリダクションは不要です")

    def apply_noise_reduction(self, reduction_method='wiener', intensity=1):
        if reduction_method == 'wiener':
            self.processed_audio = wiener(self.audio, mysize=None, noise=None)
        elif reduction_method == 'noisereduce':
            noise_clip = self.audio[0:int(0.5 * self.sr)]
            self.processed_audio = nr.reduce_noise(audio_clip=self.audio, noise_clip=noise_clip, verbose=False)

        self.processed_audio *= intensity

    def adjust_snr(self, target_snr_db):
        signal_power = np.mean(self.audio ** 2)
        noise_power = np.mean((self.audio - self.processed_audio) ** 2)
        current_snr_db = 10 * np.log10(signal_power / noise_power)

        required_snr_linear = 10 ** ((target_snr_db - current_snr_db) / 10)
        self.processed_audio *= required_snr_linear

    def parse_episode(self, output_txt):
        episodes = []
        current_episode = None
        dialogues = []
        episode_start_pattern = re.compile(r'^（(.+)が始まりました）$')
        episode_end_pattern = re.compile(r'^（(.+)は終わりました）$')
        dialogue_pattern = re.compile(r'^说话人(\d+)\s+(\d{2}:\d{2})$')

        with open(output_txt, 'r', encoding='utf-8') as file:
            text = file.read()

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            start_match = episode_start_pattern.match(line)
            if start_match:
                if current_episode is not None:
                    episodes.append(current_episode)
                current_episode = {'title': start_match.group(1), 'dialogues': []}
                continue

            end_match = episode_end_pattern.match(line)
            if end_match:
                if current_episode is not None:
                    episodes.append(current_episode)
                    current_episode = None
                continue

            if current_episode is not None:
                current_episode['dialogues'].append({
                    'speaker': '発言者',
                    'time': '時間',
                    'text': 'テキスト'
                })

            speaker_match = dialogue_pattern.match(line)
            if speaker_match:
                if dialogues:
                    current_episode['dialogues'].append({
                        'speaker': current_speaker,
                        'time': current_time,
                        'text': ' '.join(dialogues)
                    })
                    dialogues = []
                current_speaker = speaker_match.group(1)
                current_time = speaker_match.group(2)
            else:
                dialogues.append(line)

        return episodes

    def prompt_engineer(self, output_txt, output_json):
        if len(self.data) != 0:
            self.data = []

        episodes = self.parse_episode(output_txt)

        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(episodes, json_file, ensure_ascii=False, indent=4)

    def save_processed_audio(self, output_path):
        sf.write(output_path, self.processed_audio, self.sr)

    def correct_text(self, text):
        self.model.eval()
        self.model.to(self.device)

        masked_text = text.replace("間違った", "[MASK]")
        encoded_input = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            predictions = outputs.logits

        masked_index = torch.where(encoded_input["input_ids"] == self.tokenizer.mask_token_id)[1]
        predicted_id = predictions[0, masked_index].argmax(dim=-1)
        predicted_token = self.tokenizer.decode(predicted_id).strip()

        corrected_text = masked_text.replace("[MASK]", predicted_token)
        return corrected_text

    def process_audio_and_text(self, directory_path):
        # Audio processing
        self.apply_noise_reduction(reduction_method='wiener', intensity=0.5)
        self.adjust_snr(20)
        self.apply_noise_reduction_if_needed(threshold_snr=10)
        self.save_processed_audio(os.path.join(directory_path, 'processed/wav/Elements.wav'))
        print('音频处理完成')

        # Text processing
        output_txt_path = os.path.join(directory_path, 'processed/episodes_txt/')
        self.process_all_files(output_txt_path)
        combined_txt_path = os.path.join(output_txt_path, 'combined_.txt')
        self.export_to_txt(combined_txt_path)
        combined_json_path = os.path.join(output_txt_path, 'combined_.json')
        self.prompt_engineer(combined_txt_path, combined_json_path)
        print("文本处理完成")

    def correct_text_example(self):
        # Example of text correction
        asr_text = "ここで間違ったテキストが入力されます"
        corrected_text = self.correct_text(asr_text)
        print("Original Text:", asr_text)
        print("Corrected Text:", corrected_text)

def main():
    directory_path = '/path/to/your/data/'  # Update this path
    
    # Create an instance of the processor
    processor = ASRProcessor()
    
    # Start Flask server in a thread
    flask_thread = Thread(target=processor.run_flask)
    flask_thread.start()

    # Process audio and text files
    processor.process_audio_and_text(directory_path)
    
    # Demonstrate text correction
    processor.correct_text_example()

if __name__ == "__main__":
    main()
