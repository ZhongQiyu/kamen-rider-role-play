# det.py

import cv2
import mediapipe as mp
import json
import uuid
import time
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import smbus  # 用于与传感器通信 (I2C)

# 代理系统类，集成电机控制、传感器、摄像头处理和策略执行
class AgentSystem:
    def __init__(self):
        self.agent_id = str(uuid.uuid4())
        self.env = DummyVecEnv([lambda: CustomEnv()])  # RL 环境
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.sensor_module = self.init_sensor_module()  # 初始化传感器
        self.camera_module = self.init_camera_module()  # 初始化摄像头
        self.state = self.env.reset()

    # 初始化 MPU6050 传感器
    def init_sensor_module(self, bus_number=1, device_address=0x68):
        bus = smbus.SMBus(bus_number)
        bus.write_byte_data(device_address, 0x6B, 0)  # 唤醒传感器
        print("Sensor initialized")
        return bus, device_address

    # 初始化摄像头和 MediaPipe 模块
    def init_camera_module(self):
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        pose = mp_pose.Pose()
        hands = mp_hands.Hands()
        return pose, hands

    # 读取传感器数据 (使用 MPU6050)
    def read_sensor_data(self):
        bus, device_address = self.sensor_module
        accel_x = self.read_word_2c(bus, device_address, 0x3B)
        accel_y = self.read_word_2c(bus, device_address, 0x3D)
        accel_z = self.read_word_2c(bus, device_address, 0x3F)
        gyro_x = self.read_word_2c(bus, device_address, 0x43)
        gyro_y = self.read_word_2c(bus, device_address, 0x45)
        gyro_z = self.read_word_2c(bus, device_address, 0x47)
        return {
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }

    # 读取传感器 16 位数据
    def read_word_2c(self, bus, addr, reg):
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg+1)
        val = (high << 8) + low
        return val if val < 0x8000 else -((65535 - val) + 1)

    # 处理摄像头帧，进行姿态和手势识别
    def process_frame(self, frame):
        pose, hands = self.camera_module
        pose_results = pose.process(frame)
        hand_results = hands.process(frame)
        return pose_results, hand_results

    # 电机控制逻辑
    @staticmethod
    def control_motor(direction, speed):
        return {
            'type': 'motor_control',
            'direction': direction,
            'speed': speed
        }

    # 强化学习代理执行策略，结合传感器和摄像头数据
    def execute_strategy(self):
        # 读取传感器数据
        sensor_data = self.read_sensor_data()
        
        # 读取摄像头数据
        cap = cv2.VideoCapture(0)  # 打开摄像头
        ret, frame = cap.read()
        if ret:
            pose_results, hand_results = self.process_frame(frame)
        
        # 强化学习策略
        action, _states = self.model.predict(self.state)
        self.state, reward, done, info = self.env.step(action)
        
        # 基于策略选择动作
        motor_instruction = self.control_motor('forward', 50) if action == 0 else self.control_motor('backward', 50)
        instruction = {
            'agent_id': self.agent_id,
            'action': action,
            'motor_instruction': motor_instruction,
            'timestamp': int(time.time())
        }
        return instruction

    # 向服务器发送指令
    @staticmethod
    def send_instruction_to_server(instruction):
        encoded_instruction = json.dumps(instruction)
        try:
            response = requests.post(
                url="https://multi-agent-server-url.com",  # Replace with actual server URL
                data=encoded_instruction,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("success"):
                    print("Instruction executed successfully.")
                else:
                    print("Instruction execution failed.")
            else:
                print(f"Server responded with status code: {response.status_code}")
        except Exception as e:
            print(f"Failed to communicate with the server: {e}")

# 自定义环境类
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        self.action_space = gym.spaces.Discrete(2)
        self.state = [0, 0, 0, 0]

    def step(self, action):
        self.state = [0, 0, 0, 0]  # 模拟新状态
        reward = 1 if action == 0 else -1  # 奖励逻辑
        done = False  # 模拟结束条件
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = [0, 0, 0, 0]  # 重置状态
        return self.state

    def render(self, mode='human'):
        pass

# 主系统
class MainSystem:
    def __init__(self):
        self.agent_system = AgentSystem()

    def train_model(self, timesteps=10000):
        print(f"Training model for {timesteps} timesteps...")
        self.agent_system.model.learn(total_timesteps=timesteps)
        print("Training completed.")

    def run_agent(self):
        instruction = self.agent_system.execute_strategy()
        self.agent_system.send_instruction_to_server(instruction)

# 主函数
if __name__ == "__main__":
    # 初始化系统
    system = MainSystem()
    
    # 训练强化学习模型
    system.train_model(timesteps=10000)
    
    # 运行代理并发送指令
    system.run_agent()
