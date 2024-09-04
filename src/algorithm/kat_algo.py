import numpy as np
import os 
import pickle
from typing import *
import llm.llm_utils as llm
import ast
import re

NUMBER_OF_DEMONSTRATIONS = 30

def sub_sample_demonstrations(deomonstrations: List[List[int]], num_samples: int) -> List[List[int]]:
    '''
    Sub sample the demonstrations to num_samples
    '''
    def get_random_indices(list_length: int, num_samples: int) -> List[int]:
        return list(np.random.choice(range(list_length), num_samples, replace=False))

    # randomly sample the demonstrations
    random_indices: List[int] = get_random_indices(len(deomonstrations), num_samples)
    sampled_elements: List[int] = [deomonstrations[i] for i in random_indices]

    return sampled_elements

def convert_to_prompt_tokens(list: List[int]) -> List[int]:
    '''
    multiply by 100 and remove decimal places
    '''
    return [int(x * 100) for x in list]
    
def construct_demonstations():
    '''
    Construct the demonstrations for the LLM
    '''
    # load in all the demonstration from folder demonstration_runs
    demonstation_prompt: List[dict] = []

    demonstation_folder = 'demonstration_runs'

    for file in os.listdir(demonstation_folder):
        # open each folder and load in the demonstrations
        if os.path.isdir(demonstation_folder + '/' + file):
            for demo in os.listdir(demonstation_folder + '/' + file):
                if demo.endswith('.pkl'):
                    with open(demonstation_folder + '/' + file + '/' + demo, 'rb') as f:
                        data = pickle.load(f)

                        goal_tokens = convert_to_prompt_tokens(data['config']['goal'])
                        initial_positions_tokens = convert_to_prompt_tokens(data['config']['initial_position'])

                        ee_states: List[List[int]] = data['data']['ee_states']
                        ee_states_tokens = []
                        for i in range(len(ee_states)):
                            ee_states_tokens.append(convert_to_prompt_tokens(ee_states[i]))


                        demonstation_prompt.append({
                            'goal': goal_tokens,
                            'initial_position': initial_positions_tokens,
                            'ee_states': ee_states_tokens
                        })

    
    return demonstation_prompt


# save the demonstations to a txt file
def save_demonstations(demonstations: List, filename: str = 'src/tmp/demonstrations.txt'):
    '''
    Save the demonstations to a txt file
    '''
    with open(filename, 'w') as f:
        for demo in demonstations:
            f.write(f"Input: {demo['initial_position'],"->", demo['goal']}\n")
            f.write(f"Output: {demo['ee_states']}\n")
            f.write("\n")

def append_context(context: Dict, filename: str = 'src/tmp/demonstrations.txt'):
    '''
    Append the context to the demonstrations
    '''
    with open(filename, 'a') as f:
        f.write(f"{context["overview"]}\n")
        f.write(f"{context["initial_position"],"->",context["goal"]}\n")
        f.write("\n")

def generate_prompt_file(demonstrations: List, filename: str = 'src/tmp/prompt.txt', num_samples: int = 10, context: Dict = {"overview": "based on the below input predict the output as a json", "goal": "[0,0]", "initial_position": "[0,0]"}):
    '''
    Generate the prompt file for the LLM
    '''
    # sub sample the demonstrations
    # append the context to the demonstrations
    save_demonstations(demonstrations, filename)
    append_context(context, filename)
    # save the demonstrations to a file

def extract_waypoints(raw_waypoints: str) -> dict:
    '''
    Extract the waypoints from the raw waypoints
    '''
    # convert string to a list of lists 
    # Use regular expressions to find all pairs of numbers

    # get rid of waypoint starting
    print("Raw Waypoints: ", raw_waypoints)
    waypoints = re.findall(r'\[(-?\d+),\s*(-?\d+)\]', raw_waypoints)

    # Convert to list of lists with integers
    waypoints_list = [[int(x), int(y)] for x, y in waypoints]

    return_dict = {"waypoints": waypoints_list}
    return return_dict


def query_llm(starting_position: List[int], goal_position: List[int], arm_lengths: List) -> List[List[int]]:
    '''
    Query the LLM and then save the list as 
    '''
    demonstations = construct_demonstations()
    selected_demonstrations = sub_sample_demonstrations(demonstations, NUMBER_OF_DEMONSTRATIONS)


    overview = (
                f"You are a waypoint generator from the starting coordinates (first 2 values) to the goal (last 2 values). "
                f"The robot has two arms of lengths {str(arm_lengths[0])} and {str(arm_lengths[1])} and operates on an X,Y plane. "
                f"Give the trajectory to follow in JSON format with the key 'waypoints'."
                f"You can space out the waypoints"
                )
    generate_prompt_file(selected_demonstrations, 
                         num_samples=10, 
                         context={
                            #  give overview as a string of arm lengths
                            "overview": overview,
                            "goal": convert_to_prompt_tokens(goal_position), 
                            "initial_position": convert_to_prompt_tokens(starting_position)})

    
    llm_client: llm.OpenAIClient = llm.OpenAIClient()
    raw_waypoints: str = llm_client.process_test("src/tmp/prompt.txt")

    waypoints = extract_waypoints(raw_waypoints)

    # convert from token space to the original space
    for i in range(len(waypoints["waypoints"])):
        waypoints["waypoints"][i] = [x / 100 for x in waypoints["waypoints"][i]]

    return waypoints["waypoints"]

def up_sample_waypoints(waypoints: List[List[int]], factor: int) -> List[List[int]]:
    '''
    Make every occuring value appear 10 times in order
    '''
    result = []
    for sublist in waypoints:
        for _ in range(factor):
            result.append(sublist.copy())  # use copy to avoid reference issues
    return result


def kat_algo(env, current_state) -> Tuple[float, float]:
    if not hasattr(env, 'llm'):
        waypoints = query_llm(
            starting_position=current_state,
            goal_position=env.goal,
            arm_lengths=[env.l1, env.l2]
        )
        print("Waypoints: ", waypoints)
        print("Waypoint Length: ", len(waypoints))
        env.all_waypoints = waypoints

        # increase the number of waypoints to allow for execution time

        env.waypoints = up_sample_waypoints(waypoints, 5)
        
        env.llm = True
    # print("Queried")

    # get the next waypoint
    if len(env.waypoints) == 0:
        return np.array([0, 0])
    
    next_waypoint = env.waypoints.pop(0)

    # calculate the action using inverse kinematics
    goal_theta1, goal_theta2 = env.inverse_kinematics(*next_waypoint)
    
    # Calculate the actions needed to move towards the goal joint angles
    action1 = goal_theta1 - env.theta1
    action2 = goal_theta2 - env.theta2

    return np.array([action1, action2])
