import sys
import importlib
from pathlib import Path
import json
import random
import os
import moviepy.editor as mp
import numpy as np
import pickle

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent
sys.path.append(str(src_path))

from env.simple_robot_arm_env import SimpleRobotArmEnv
from utils.file_io import load_config, save_data, create_folder_if_not_exists



def generate_configurations(config: dict, number_to_generate: int, custom_output_suffix: str = "") -> list[dict]:
    '''
    Generate the configurations for the LLM
    '''

    configuration_list = []
    for i in range(number_to_generate):
        configuration = {
            "action_max_min": config['base_config']['action_max_min'],
            "l1": config['base_config']['l1'],
            "l2": config['base_config']['l2'],
            "max_steps": config['base_config']['max_steps'],
            "goal": [
                random.uniform(config['base_config']['goal']['random_range']['x'][0], config['base_config']['goal']['random_range']['x'][1]),
                random.uniform(config['base_config']['goal']['random_range']['y'][0], config['base_config']['goal']['random_range']['y'][1])
            ],
            "initial_position": [
                random.uniform(config['base_config']['initial_position']['random_range']['x'][0], config['base_config']['initial_position']['random_range']['x'][1]),
                random.uniform(config['base_config']['initial_position']['random_range']['y'][0], config['base_config']['initial_position']['random_range']['y'][1])
            ],
            "output_folder": config['base_config']['output_folder'] + f"/{custom_output_suffix}_run_{i}",
            "algorithm": config['base_config']['algorithm']
        }
        configuration_list.append(configuration)
    return configuration_list

def capture_data(env, config):
    print("Saving data: ", config['output_folder'])
    save_data({
        "config": config,
        "data": {
            "joint_states": env.joint_states,
            "ee_states": env.ee_states,
            "rewards": env.rewards,
            "dones": env.dones,
            "actions": env.actions,

            "ensomble": env.ensomble_dictionary
        },
        "sucessful": env.dones[-1]
    }, config['output_folder'] + '/output.pkl')

    env.render_to_video(config['output_folder'] + '/robot_arm_animation.mp4')



def test(config: dict, get_action: callable, dictionary: dict = None, waypoints: list[list[int]] = None):
    # load in the environment
    env = SimpleRobotArmEnv(config)
    state = env.reset()
    done = False

    env.joint_states = []
    env.ee_states = []
    env.rewards = []
    env.dones = []
    env.actions = []

    env.current_algorithm_name = get_action.__name__

    if dictionary is not None:
        env.ensomble_dictionary = dictionary
    else:
        env.ensomble_dictionary = None

    if waypoints is not None:
        env.all_waypoints = waypoints
    
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
    
    capture_data(env, config)

    output_dict = {
        "initial_position": config['initial_position'],
        "goal": config["goal"],
        "ee_states": env.ee_states,
    }
    success = env.dones[-1]

    return output_dict, success


def up_sample_waypoints(waypoints: list[list[int]], factor: int) -> list[list[int]]:
    '''
    Make every occuring value appear 10 times in order
    '''
    result = []
    for sublist in waypoints:
        for _ in range(factor):
            result.append(sublist.copy())  # use copy to avoid reference issues
    return result
    

if __name__ == '__main__':

    # load in the config file
    config = load_config('src/config/train_cp_ensomble.json')

    # set up the algorithm
    algorithm = config['base_config']['algorithm']
    algorithm_module = importlib.import_module(f'algorithm.{algorithm}')
    algorithm = getattr(algorithm_module, algorithm)
    # initialise the algorithm and create an instace
    algorithm = algorithm(config)

    algorithm.set_up(config)

    configuration_training: list = generate_configurations(config, config["training_config"]["number_of_requestable_demonstrations"], custom_output_suffix="/training/")

    for run_number in range(len(configuration_training)):
        print("*********************************************")
        print(f"Training configuration {run_number}")
        # run the algorithm
        training_dictionary: dict = algorithm.train(configuration_training[run_number])
        
        if training_dictionary['help_needed'] == True:
            print("Help is needed")
            # roll out with inverse kinematics
            get_action = algorithm.inverse_kinematics
            # modify the configuration to include the training dictionary
            demonstration, _ = test(configuration_training[run_number], get_action, dictionary=training_dictionary)
            algorithm.add_requested_demonstrations(demonstration)

        else:
            print("Help is not needed")
            # roll out with the algorithm
            get_action = algorithm.get_action
            
            # find the best prediction
            best_prediction = training_dictionary["pathways"][0]
            for i in range(1, len(training_dictionary["pathways"])):
                if training_dictionary["pathways"][i]["logprob"] > best_prediction["logprob"]:
                    best_prediction = training_dictionary["pathways"][i]

            algorithm.waypoints_to_execute = up_sample_waypoints(best_prediction["waypoint_trajectory"], 5)
            _, _ = test(configuration_training[run_number], get_action, dictionary=training_dictionary, waypoints=best_prediction["waypoint_trajectory"])
            algorithm.waypoints_to_execute = None

    #  testing 
    configuration_testing: list = generate_configurations(config, config["testing_runs"], custom_output_suffix="/testing/")
    sucess = 0
    
    # pickle save the configuration_testing
    # with open('testing_configurations.pkl', 'wb') as f:
    #     pickle.dump(configuration_testing, f)

    # # load the configuration_testing
    # with open('testing_configurations.pkl', 'rb') as f:
    #     configuration_testing = pickle.load(f)

    for run_number in range(len(configuration_testing)):
        print("*********************************************")
        print(f"Testing configuration {run_number}")
        # roll out with the algorithm
        test_dictionary: dict = algorithm.test(configuration_testing[run_number])

        get_action = algorithm.get_action
        algorithm.waypoints_to_execute = up_sample_waypoints(test_dictionary['waypoint_trajectory'], 5)
        run_summary, done = test(configuration_testing[run_number], get_action, waypoints=test_dictionary["waypoint_trajectory"])

        if done:
            sucess += 1
    
    print("===============================================")
    print(f"Sucess rate: {sucess}/{len(configuration_testing)}")
    print("===============================================")

    # capture data from the algorithum 