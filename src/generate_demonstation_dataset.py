import sys
import importlib
from pathlib import Path
import json
import random
import os
import moviepy.editor as mp


SUCSESS_DEMOS = 0
# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent
sys.path.append(str(src_path))

from env.simple_robot_arm_env import SimpleRobotArmEnv
from utils.file_io import load_config, save_data, create_folder_if_not_exists

def load_generate_demonstration_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
def summary_video(demonstration_folder):
    # Initialize an empty list to store video clips
    clips = []

    # Traverse the demonstration folder and subfolders to find .mp4 files
    for file in os.listdir(demonstration_folder):
        file_path = os.path.join(demonstration_folder, file)
        print(file_path)
        if os.path.isdir(file_path):
            for demo in os.listdir(file_path):
                demo_path = os.path.join(file_path, demo)
                print("==>",demo_path)
                if demo.endswith('.mp4'):
                    print("=======>",demo_path)
                    clips.append(mp.VideoFileClip(demo_path))

    # Concatenate the video clips
    final_clip = mp.concatenate_videoclips(clips)

    # Write the result to a file
    output_path = os.path.join(demonstration_folder, "combined_video.mp4")
    final_clip.write_videofile(output_path, codec="libx264")
    
def demonstration_summary(demonstration_folder):
    '''
    Summarize the demonstrations in the folder
    '''
    # open each folder and load in the demonstration video 
    # and append all the videos together to create a summary
    summary_video(demonstration_folder)

def generate_random_config(run_number, config_ranges, output_parent_folder):
    goal_x_range = config_ranges['goal_range']['x']
    goal_y_range = config_ranges['goal_range']['y']
    init_x_range = config_ranges['initial_position_range']['x']
    init_y_range = config_ranges['initial_position_range']['y']

    config = {
        "action_max_min": 0.5,
        "l1": 1.0,
        "l2": 1.0,
        "goal": [random.uniform(goal_x_range[0], goal_x_range[1]), random.uniform(goal_y_range[0], goal_y_range[1])],
        "max_steps": 500,
        "initial_position": [random.uniform(init_x_range[0], init_x_range[1]), random.uniform(init_y_range[0], init_y_range[1])],
        "output_folder": f"{output_parent_folder}/run_ik_{run_number}",
        "algorithm": "inverse_kinematics"
    }
    return config

def main(run_number: int, config_ranges: dict, output_parent_folder: str):
    # Generate random configuration
    config = generate_random_config(run_number, config_ranges, output_parent_folder)
    create_folder_if_not_exists(config['output_folder'])

    # Save the generated config to a file
    config_path = Path(output_parent_folder+ "/run_ik_" + str(run_number) + "/config_run_" + str(run_number) + ".json")
    with config_path.open('w') as config_file:
        json.dump(config, config_file, indent=4)

    # Create the environment with the generated configuration
    env = SimpleRobotArmEnv(config)
    env.current_algorithm_name = config["algorithm"]

    # Dynamically import the specified algorithm
    algorithm_module = importlib.import_module(f'algorithm.{config["algorithm"]}')
    get_action = getattr(algorithm_module, config["algorithm"])

    # Run a test and save the video
    state = env.reset()
    done = False

    env.joint_states = []
    env.ee_states = []
    env.rewards = []
    env.dones = []
    env.actions = []

    # Collect frames to reach the goal using the selected algorithm
    for _ in range(env.max_steps):
        if done:
            break
        action = get_action(env, state)
        ee_state, joint_position_state, reward, done = env.step(action)

        env.joint_states.append(joint_position_state)
        env.ee_states.append(ee_state)
        env.rewards.append(reward)
        env.dones.append(done)
        env.actions.append(action)

        env.all_waypoints.append(ee_state)

        # print(f"Action: {action}, State: {ee_state}, Reward: {reward}")
    


    # Save the collected data
    save_data({
        "config": config,
        "data": {
            "joint_states": env.joint_states,
            "ee_states": env.ee_states,
            "rewards": env.rewards,
            "dones": env.dones,
            "actions": env.actions,

            # "ensomble": env.ensomble_dictionary
        },
        "sucessful": env.dones[-1]
    }, config['output_folder'] + '/output.pkl')

    # Render the collected actions to a video file
    env.render_to_video(config['output_folder'] + '/robot_arm_animation.mp4')

    # Replay the demonstration from the saved data
    # env.replay_demonstration(config['output_folder'] + '/output.pkl')
    

    if env.dones[-1]:
        global SUCSESS_DEMOS
        SUCSESS_DEMOS += 1
    else:
        print(f"Run {run_number} failed")

if __name__ == '__main__':
    generate_config_path = src_path / 'config/generate_demonstration_dataset.json'
    generate_config = load_generate_demonstration_config(generate_config_path)
    num_runs = generate_config['num_runs']
    config_ranges = generate_config
    output_parent_folder = generate_config['output_parent_folder']

    for run_number in range(1, num_runs + 1):
        main(run_number, config_ranges, output_parent_folder)
    
    print(f"Number of successful demonstrations: {SUCSESS_DEMOS} out of {num_runs}")
    demonstration_summary(output_parent_folder)