# ppo.py

import tensorflow as tf
import horovod.tensorflow as hvd
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# -----------------------------------
# Part 1: TensorFlow Distributed Training Setup
# -----------------------------------

def setup_tensorflow_distributed():
    # Initialize Horovod
    hvd.init()

    # Configure GPU settings for distributed training
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Define a simple neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Wrap the optimizer with Horovod's Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(tf.optimizers.Adam(0.001 * hvd.size()))

    # Compile the model for distributed training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Generate a random dataset for training
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([1000, 10]), tf.random.uniform([1000], maxval=10, dtype=tf.int32))
    ).batch(32 * hvd.size())

    # Use callbacks to synchronize all processes
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint.h5'))

    # Train the model
    model.fit(dataset, epochs=5, callbacks=callbacks)

# -----------------------------------
# Part 2: Ray RLlib Single Agent Setup
# -----------------------------------

def setup_ray_rllib_single_agent():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define the training configuration using a simple reinforcement learning environment
    config = {
        "env": "CartPole-v0",  # This environment will simulate a single agent
        "num_workers": 4,
        "framework": "tf",
        "train_batch_size": 4000,
        "gamma": 0.99,  # Discount factor for future rewards
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),  # Learning rate search
        "model": {
            "fcnet_hiddens": [256, 256],  # Neural network architecture
            "fcnet_activation": "relu",
        }
    }

    # Use the Proximal Policy Optimization (PPO) algorithm for training
    results = tune.run(
        "PPO",
        config=config,
        stop={"episode_reward_mean": 200},  # Stopping criteria
        verbose=1
    )

    # Output the best training result
    print("Best PPO result:", results.get_best_result())

# -----------------------------------
# Main Execution
# -----------------------------------

if __name__ == "__main__":
    # Execute TensorFlow Distributed Training
    setup_tensorflow_distributed()

    # Execute Ray RLlib Single Agent Training
    setup_ray_rllib_single_agent()
