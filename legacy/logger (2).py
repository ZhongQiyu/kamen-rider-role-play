#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import re
import gc
import fasttext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum, auto

# Logger Management
class LoggerManager:
    def __init__(self, log_directory='logs'):
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        self.logger = logging.getLogger('kamen_rider_blade')
        self.logger.setLevel(logging.DEBUG)

        levels = {'app.log': logging.INFO, 'error.log': logging.ERROR, 'debug.log': logging.DEBUG}
        for filename, level in levels.items():
            handler = logging.FileHandler(os.path.join(log_directory, filename))
            handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.error_logger = logging.getLogger('error_logger')
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(os.path.join(log_directory, 'error.log'))
        error_handler.setFormatter(formatter)
        self.error_logger.addHandler(error_handler)

    def get_logger(self):
        return self.logger

    def get_error_logger(self):
        return self.error_logger

# Command Enum and Utility
class Command(Enum):
    CONVERT_REPOS = auto()
    LOGOUT_USERS = auto()
    PROCESS_AUDIO = auto()
    UNKNOWN = auto()

class UsageError(Exception):
    pass

def usage():
    print("Usage: script.py command [options]")
    print("Commands:")
    print("  convert_repos PARENT_REPO_PATH EXTERNAL_REPOS_DIR  Convert external repos to submodules or subtrees.")
    print("  logout_users                                      Log off all users.")
    print("  process_audio                                     Process audio files in the specified folders.")
    sys.exit(1)

# Repo Converter
class RepoConverter:
    def __init__(self, parent_repo_path, external_repos_dir):
        self.parent_repo_path = parent_repo_path
        self.external_repos_dir = external_repos_dir

    def convert_repos(self):
        os.chdir(self.parent_repo_path)

        for repo_dir in os.listdir(self.external_repos_dir):
            repo_dir_path = os.path.join(self.external_repos_dir, repo_dir)
            if os.path.isdir(os.path.join(repo_dir_path, ".git")):
                repo_url = subprocess.getoutput(f"git -C {repo_dir_path} config --get remote.origin.url")
                if not repo_url:
                    print(f"ERROR: Unable to find remote url for {repo_dir_path}")
                    continue

                repo_path = repo_dir_path[len(self.parent_repo_path) + 1:]

                subprocess.run(["git", "rm", "-rf", repo_path])
                subprocess.run(["rm", "-rf", repo_path])

                subprocess.run(["git", "submodule", "add", repo_url, repo_path])
            else:
                print(f"WARNING: {repo_dir_path} is not a git repository.")

        subprocess.run(["git", "commit", "-m", "Transformed external repos to submodules or subtrees."])
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"])

# User Management
class UserManager:
    def __init__(self):
        pass

    def logout_users(self):
        if os.getuid() != 0:
            print("This script must be run as root", file=sys.stderr)
            sys.exit(1)

        users = subprocess.getoutput("cut -d: -f1 /etc/passwd").splitlines()
        for user in users:
            subprocess.run(["pkill", "-KILL", "-u", user])

        print("All users have been logged off.")

# Audio Processor
class AudioProcessor:
    def __init__(self, input_folder, intermediate_folder, output_folder):
        self.input_folder = input_folder
        self.intermediate_folder = intermediate_folder
        self.output_folder = output_folder

        os.makedirs(self.intermediate_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

    def process_audio_files(self):
        files = [f for f in os.listdir(self.input_folder) if f.endswith(('.m4a', '.mp3'))]
        if not files:
            print(f"No audio files found in {self.input_folder}")
            sys.exit(1)

        for input_file in files:
            input_path = os.path.join(self.input_folder, input_file)
            base_name, extension = os.path.splitext(input_file)
            base_name = base_name.replace('.', '_')

            print(f"Processing {input_file} with extension {extension}")

            pcm_path = os.path.join(self.intermediate_folder, f"{base_name}.pcm")
            subprocess.run(["ffmpeg", "-i", input_path, "-f", "s16le", "-acodec", "pcm_s16le", pcm_path])
            print(f"Converted to PCM: {pcm_path}")

            denoised_pcm_path = os.path.join(self.intermediate_folder, f"{base_name}_denoised.pcm")
            subprocess.run(["./examples/rnnoise_demo", pcm_path, denoised_pcm_path])
            print(f"Noise reduction applied: {denoised_pcm_path}")

            output_m4a_path = os.path.join(self.output_folder, f"{base_name}_denoised.m4a")
            subprocess.run(["ffmpeg", "-f", "s16le", "-ar", "44100", "-ac", "1", "-i", denoised_pcm_path, output_m4a_path])
            print(f"Converted {denoised_pcm_path} to {output_m4a_path} as .m4a")

            os.remove(pcm_path)
            os.remove(denoised_pcm_path)

# Data Processing
class DataProcessor:
    def __init__(self, data_folder_pc, data_folder_tp, output_combined_file):
        self.data_folder_pc = data_folder_pc
        self.data_folder_tp = data_folder_tp
        self.output_combined_file = output_combined_file

    def concatenate_files(self, directory, output_file):
        with open(output_file, 'w', encoding='utf-8') as outfile:
            pass

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith('.txt') and not filename.startswith('.'):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        outfile.write(content + '\n')
        print(f"All txt files have been concatenated into {output_file}")

    def process_text_files(self, folder, output_file):
        directory = f'/mnt/c/Users/xiaoy/Documents/Yuki/data/complete/{folder}'
        data = []

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                tieba_name_match = re.search(fr'{folder}(.*){folder}', filename)
                tieba_name = tieba_name_match.group(1) if tieba_name_match else ''
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split('\t\t')
                        if len(parts) == 2:
                            ask, answer = parts
                            data.append({
                                'data_folder': folder,
                                'tieba_name': tieba_name,
                                'ask_content': ask,
                                'answer_content': answer
                            })

        df_combined = pd.DataFrame(data)
        df_combined.to_csv(output_file, sep='\t', index=False, header=True)
        print(f'File processing completed, results saved to {output_file}')

    def run_all(self):
        pc_output_file = os.path.join(self.data_folder_pc, 'pc_all.txt')
        tp_output_file = os.path.join(self.data_folder_tp, 'tp_all.txt')
        combined_file = self.output_combined_file

        self.concatenate_files(self.data_folder_pc, pc_output_file)
        self.process_text_files('tp', tp_output_file)
        self.combine_files(pc_output_file, tp_output_file, combined_file)

# Main Execution
def parse_command(args):
    if len(args) < 1:
        return Command.UNKNOWN, []

    command_str = args[0].lower()
    if command_str == "convert_repos":
        return Command.CONVERT_REPOS, args[1:]
    elif command_str == "logout_users":
        return Command.LOGOUT_USERS, args[1:]
    elif command_str == "process_audio":
        return Command.PROCESS_AUDIO, args[1:]
    else:
        return Command.UNKNOWN, []

def main():
    command, args = parse_command(sys.argv[1:])

    if command == Command.CONVERT_REPOS:
        if len(args) != 2:
            raise UsageError("convert_repos requires PARENT_REPO_PATH and EXTERNAL_REPOS_DIR")
        repo_converter = RepoConverter(args[0], args[1])
        repo_converter.convert_repos()

    elif command == Command.LOGOUT_USERS:
        user_manager = UserManager()
        user_manager.logout_users()

    elif command == Command.PROCESS_AUDIO:
        if len(args) != 3:
            raise UsageError("process_audio requires INPUT_FOLDER, INTERMEDIATE_FOLDER, and OUTPUT_FOLDER")
        audio_processor = AudioProcessor(args[0], args[1], args[2])
        audio_processor.process_audio_files()

    else:
        usage()

if __name__ == "__main__":
    try:
        main()
    except UsageError as e:
        print(f"Error: {e}", file=sys.stderr)
        usage()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
