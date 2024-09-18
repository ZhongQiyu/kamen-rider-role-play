# logger.py

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

# File and Directory Check Functions
def check_file_exists(file_path):
    if os.path.isfile(file_path):
        print(f"找到文件 {file_path}")
        # 在这里执行相应的命令
    else:
        print(f"未找到文件 {file_path}")
        # 在这里执行其他命令

def check_directory_exists(directory_path):
    if os.path.isdir(directory_path):
        print(f"找到目录 {directory_path}")
        # 在这里执行相应的命令
    else:
        print(f"未找到目录 {directory_path}")
        # 在这里执行其他命令

# Repo Converter
class RepoConverter:
    def __init__(self, parent_repo_path, external_repos_dir):
        self.parent_repo_path = parent_repo_path
        self.external_repos_dir = external_repos_dir

    def convert_repos(self):
        check_directory_exists(self.parent_repo_path)
        check_directory_exists(self.external_repos_dir)
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
        check_directory_exists(self.input_folder)
        check_directory_exists(self.intermediate_folder)
        check_directory_exists(self.output_folder)
        
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
    # DataProcessor code unchanged...

# Main Execution
def parse_command(args):
    # parse_command code unchanged...

def main():
    # main function code unchanged...

if __name__ == "__main__":
    try:
        main()
    except UsageError as e:
        print(f"Error: {e}", file=sys.stderr)
        usage()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
