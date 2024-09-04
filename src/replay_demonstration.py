import sys
import importlib
import argparse
from pathlib import Path
import numpy as np
from env.simple_robot_arm_env import SimpleRobotArmEnv
from utils.file_io import save_data, load_data, create_folder_if_not_exists

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent
sys.path.append(str(src_path))

def main(input_file):
    # Load previously generated run data
    previous_run_data = load_data(input_file)
    previous_config = previous_run_data['config']
    previous_data = previous_run_data['data']

    # Create the environment with the loaded configuration from the previous run
    env = SimpleRobotArmEnv(previous_config)

    # Dynamically import the specified algorithm
    algorithm_module = importlib.import_module(f'algorithm.{previous_config["algorithm"]}')
    get_action = getattr(algorithm_module, previous_config["algorithm"])

    # Initialize the robot to the same starting positions
    env.theta1, env.theta2 = previous_data['joint_states'][0]
    state = np.array(env.forward_kinematics(env.theta1, env.theta2)[-1])
    done = False

    env.joint_states = []
    env.ee_states = []
    env.rewards = []
    env.dones = []
    env.actions = []

    # Execute the same actions
    for action in previous_data['actions']:
        if done:
            break
        ee_state, joint_position_state, reward, done = env.step(action)

        env.joint_states.append(joint_position_state)
        env.ee_states.append(ee_state)
        env.rewards.append(reward)
        env.dones.append(done)
        env.actions.append(action)

        print(f"Action: {action}, State: {ee_state}, Reward: {reward}")

    # Create output folder if it doesn't exist
    create_folder_if_not_exists(previous_config['output_folder'])


    # Render the new collected actions to a video file
    new_video_filename = 'replay_robot_arm_animation.mp4'
    env.render_to_video(new_video_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a previously generated robot arm run.')
    parser.add_argument('input_file', type=str, help='Path to the input pickle file.')
    args = parser.parse_args()

    main(args.input_file)
