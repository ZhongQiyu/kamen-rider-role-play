# char_control.py

import json
import uuid
import time
import requests

# Function to control motor direction and speed
def control_motor(direction, speed):
    # Construct the instruction
    instruction = {
        'type': 'motor_control',
        'direction': direction,
        'speed': speed
    }
    # Return the instruction
    return instruction

# Function to send instruction to server for deployment
def send_instruction_to_server(instruction):
    # Convert instruction to JSON
    encoded_instruction = json.dumps(instruction)
    
    try:
        response = requests.post(
            url="https://multi-agent-server-url.com",  # Replace with actual server URL
            data=encoded_instruction,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            # Parse the server response
            response_data = response.json()
            if response_data.get("success"):
                print("Instruction executed successfully.")
            else:
                print("Instruction execution failed.")
        else:
            print(f"Server responded with status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to communicate with the server: {e}")

# Function to generate UUID for each agent
def generate_uuid():
    return str(uuid.uuid4())

# Define a class for the agent
class Agent:
    def __init__(self, agent_id, strategy):
        self.agent_id = agent_id
        self.strategy = strategy
        self.state = {}
    
    # Agent decision-making based on strategy and state
    def execute_strategy(self):
        # Example: Apply strategy to the agent's current state
        action = self.strategy(self.state)
        
        # Agent performs motor control based on the strategy
        if action == 'move_forward':
            motor_instruction = control_motor('forward', 50)
        else:
            motor_instruction = control_motor('stop', 0)  # Example stop or wait action
        
        instruction = {
            'agent_id': self.agent_id,
            'action': action,
            'motor_instruction': motor_instruction,
            'timestamp': int(time.time())
        }
        return instruction

# Example strategy function
def basic_strategy(state):
    # Example decision-making logic: agent decides to move based on state
    if state.get('energy', 100) > 50:
        return 'move_forward'
    else:
        return 'wait'

# Main function for multi-agent system control
if __name__ == "__main__":
    # Create multiple agents with different strategies
    agent1 = Agent(generate_uuid(), basic_strategy)
    agent2 = Agent(generate_uuid(), basic_strategy)
    
    # Simulate agent state changes and decision-making
    agent1.state = {'energy': 60}  # Example state for agent1
    agent2.state = {'energy': 40}  # Example state for agent2
    
    # Agents execute their strategies
    instruction1 = agent1.execute_strategy()
    instruction2 = agent2.execute_strategy()
    
    # Send the instructions for deployment
    send_instruction_to_server(instruction1)
    send_instruction_to_server(instruction2)
