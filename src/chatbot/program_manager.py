# program_manager.py

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

class ProgramManager:
    def __init__(self, project_structure_path, dialogue_data_path, command=None, nodes_array=None, head_node_ip=None, port=None, video_dir=None, split_size_mb=None):
        self.project_structure_path = Path(project_structure_path)
        self.dialogue_data_path = Path(dialogue_data_path)
        self.project_structure = self.load_json(self.project_structure_path)
        self.dialogues = self.load_json(self.dialogue_data_path)
        self.command = command
        self.nodes_array = nodes_array
        self.head_node_ip = head_node_ip
        self.port = port
        self.video_dir = video_dir
        self.split_size = split_size_mb * 1024 * 1024 if split_size_mb else None

    def load_json(self, path):
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)

    def save_json(self, data, path):
        with path.open('w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def run_command(self, command):
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Command '{command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command '{command}'. Error: {e}")

    def manage_cluster(self, action):
        if action == "start":
            print(f"Starting Ray head node on {self.head_node_ip}...")
            subprocess.Popen([
                'ray', 'start', '--head',
                f'--node-ip-address={self.head_node_ip}',
                f'--port={self.port}', '--block'
            ])

            for node in self.nodes_array[1:]:
                print(f"Starting Ray worker on {node}...")
                subprocess.Popen([
                    'ray', 'start', f'--address={self.head_node_ip}:{self.port}', '--block'
                ])
            time.sleep(30)  # Wait for nodes to start
            print("Ray cluster started.")
        elif action == "stop":
            print("Stopping Ray cluster...")
            subprocess.run(['ray', 'stop'], check=True)

    def copy_files(self, src_path, dest_path, file_extension):
        print(f"Copying files from {src_path} to {dest_path}...")
        os.makedirs(dest_path, exist_ok=True)
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith(file_extension):
                    shutil.copy(os.path.join(root, file), dest_path)
        print("All files have been copied.")

    def split_large_files(self):
        print(f"Splitting large files in {self.video_dir}...")
        for file_path in Path(self.video_dir).glob("*.mov"):
            file_size = file_path.stat().st_size
            if file_size > self.split_size:
                print(f"Splitting file: {file_path}")
                subprocess.run([
                    'split', '-b', f"{self.split_size}m", str(file_path),
                    f"{file_path.stem}_part_"
                ], check=True)
            else:
                print(f"File {file_path} is smaller than the split size threshold, skipping.")
        print("All file splits completed.")

# Usage example:
if __name__ == "__main__":
    manager = IntegratedManager('path/to/project_structure.json', 'path/to/dialogues.json', nodes_array=["node1", "node2"], head_node_ip="192.168.0.1", port=6379, video_dir="data/video", split_size_mb=50)
    manager.manage_cluster("start")
    manager.run_command("your_command_here")
    manager.copy_files("/source/path", "/destination/path", ".pdf")
    manager.split_large_files()
    manager.save_json(manager.dialogues, manager.dialogue_data_path)
