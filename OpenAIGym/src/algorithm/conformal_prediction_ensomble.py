import numpy as np
import os 
import pickle
from typing import *
import llm.llm_utils as llm
import ast
import re
import shutil
import time

import algorithm.inverse_kinematics as InverseKinematics


class conformal_prediction_ensomble:
    def __init__(self, config):
        print("Initialising conformal prediction ensomble")
        self.llm_queried: bool  = False
        self.number_of_default_demonstrations: int = config['training_config']['defualt_demonstrations']['number_of_defualt_demonstrations']
        self.number_of_requestable_demonstrations: int = config['training_config']['number_of_requestable_demonstrations']
        self.config = config

        self.l1 = config['base_config']['l1']
        self.l2 = config['base_config']['l2']

    #         "ensoemble_config": {
    #     "number_of_ensomble_models": 10,
    #     "temperature": 0.1,
    #     "delta": 0.4
    # },
        
        self.temperture = config["ensoemble_config"]["temperature"]
        self.number_of_ensomble_trajectories = config["ensoemble_config"]["number_of_ensomble_models"]
        self.delta = config["ensoemble_config"]["delta"]

        self.llm_client: llm.OpenAIClient = llm.OpenAIClient()
        self._requested_demonstrations = []
    
    def set_up(self, config: dict):
        # set up with initial set of demonstrations
        self.assign_demonstrations()
        self.calibrate()
        return
        # raise NotImplementedError("Subclasses must implement set_up method")

    def inverse_kinematics(self, env, current_state):
        return InverseKinematics.inverse_kinematics(env, current_state)
    
    
    def _inverse_kinematics(self, x, y):
        '''
        does not require env to compute
        '''
        # Calculate cos(theta2) using the cosine law
        cos_theta2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)  # Clip to the valid range [-1, 1] to avoid numerical issues
        
        # Calculate sin(theta2) carefully to handle potential numerical instability
        sin_theta2 = np.sqrt(max(0, 1 - cos_theta2**2))  # Ensure the argument is non-negative
        
        # Calculate theta2
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        
        # Compute k1 and k2 for theta1 calculation
        k1 = self.l1 + self.l2 * cos_theta2
        k2 = self.l2 * sin_theta2
        
        # Calculate theta1 using the inverse tangent
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        # Normalize angles to the range [-π, π]
        theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
        
        return theta1, theta2

    def get_action(self, env, current_state):
        if len(self.waypoints_to_execute) == 0:
            return np.array([0, 0])
        
        next_waypoint = self.waypoints_to_execute.pop(0)

        # calculate the action using inverse kinematics
        goal_theta1, goal_theta2 = self._inverse_kinematics(*next_waypoint)
        
        # Calculate the actions needed to move towards the goal joint angles
        action1 = goal_theta1 - env.theta1
        action2 = goal_theta2 - env.theta2

        return np.array([action1, action2])

    def _save_demonstations(self, demonstations: List[dict], filename: str = 'src/tmp/_demonstrations.txt'):
        '''
        Save the demonstations to a txt file
        '''


        print("Total number of demonstrations: ", len(demonstations + self._requested_demonstrations))
        
        with open(filename, 'w') as f:
            for demo in demonstations:
                f.write("Input:" + str(demo['initial_position']) + "->" + str(demo['goal']))
                f.write("\n")
                f.write(f"Output: {demo['ee_states']}\n")
                f.write("\n")

            for demo in self._requested_demonstrations:
                f.write("Input:" + str(demo['initial_position']) + "->" + str(demo['goal']))
                f.write("\n")
                f.write(f"Output: {demo['ee_states']}\n")
                f.write("\n")

        return filename

    def _append_context(self,context: Dict, in_filepath: str, out_filename: str = 'src/tmp/_demonstrations.txt'):
        '''
        Append the context to the demonstrations
        '''
        # print (f"{context["initial_position"],"->",context["goal"]}\n")
        print((f"{context["initial_position"],"->",context["goal"]}\n"))

        # copy the file to a new file
        shutil.copy(in_filepath, out_filename)

        with open(out_filename, 'a') as f:
            f.write(f"{context["overview"]}\n")
            f.write("Input:" + str(context['initial_position']) + "->" + str(context['goal']))
            f.write("\n")

        return out_filename
    
    def add_requested_demonstrations(self, demonstrations: List[dict]):
        '''
        Add requested demonstrations to the list of demonstrations
        '''

        # iterate through the demonstrations and convert them to prompt tokens
        for i in range(len(demonstrations["ee_states"])):
            demonstrations["ee_states"][i] = self._convert_to_prompt_tokens(demonstrations["ee_states"][i])
    

        demonstrations["goal"] = self._convert_to_prompt_tokens(demonstrations["goal"])
        demonstrations["initial_position"] = self._convert_to_prompt_tokens(demonstrations["initial_position"])

        self._requested_demonstrations.append(demonstrations)
    

    def _sub_sample_demonstrations(self, deomonstrations: List[List[int]], num_samples: int, demonstration_indexes: List = None) -> List[List[int]]:
        '''
        Sub sample the demonstrations to num_samples
        '''
        def get_random_indices(list_length: int, num_samples: int) -> List[int]:
            return list(np.random.choice(range(list_length), num_samples, replace=False))

        # randomly sample the demonstrations
        if demonstration_indexes == None:
            random_indices: List[int] = get_random_indices(len(deomonstrations), num_samples)
            demonstration_indexes = random_indices
        sampled_elements: List[int] = [deomonstrations[i] for i in demonstration_indexes]

        return sampled_elements

    def _convert_to_prompt_tokens(self,list: List[int]) -> List[int]:
        '''
        multiply by 100 and remove decimal places
        '''

        return [int(x * 100) for x in list]

    def _construct_demonstations(self) -> List[dict]:
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

                            goal_tokens = self._convert_to_prompt_tokens(data['config']['goal'])
                            initial_positions_tokens = self._convert_to_prompt_tokens(data['config']['initial_position'])

                            ee_states: List[List[int]] = data['data']['ee_states']
                            ee_states_tokens = []
                            for i in range(len(ee_states)):
                                ee_states_tokens.append(self._convert_to_prompt_tokens(ee_states[i]))


                            demonstation_prompt.append({
                                'goal': goal_tokens,
                                'initial_position': initial_positions_tokens,
                                'ee_states': ee_states_tokens
                            })

        
        return demonstation_prompt
    
    def _generate_prompt_file(self, demonstrations: List, filename: str = 'src/tmp/_prompt.txt', num_samples: int = 10, context: Dict = {"overview": "based on the below input predict the output as a json", "goal": "[0,0]", "initial_position": "[0,0]"}):
        '''
        Generate the prompt file for the LLM
        '''
        # sub sample the demonstrations
        # append the context to the demonstrations
        demonstration_filepath = self._save_demonstations(demonstrations,filename)
        demonstration_context_filepath = self._append_context(context, in_filepath=demonstration_filepath, out_filename="src/tmp/_context.txt")

        return demonstration_filepath, demonstration_context_filepath

    def assign_demonstrations(self):
        all_demonstations = self._construct_demonstations()
        selected_demonstrations = self._sub_sample_demonstrations(all_demonstations,
                                                                  num_samples=self.number_of_default_demonstrations, 
                                                                  demonstration_indexes=self.config['training_config']['defualt_demonstrations']['default_demonstrations_ids'])

        print("Selected demonstrations:")
        self.selected_demonstrations = selected_demonstrations

    def _query_llm(self, starting_position: List[int], goal_position: List[int], arm_lengths: List) -> Tuple[List[List[int]], dict]:
        '''
        Query the LLM and then save the list as 
        '''
        


        overview = (
                    f"You are a waypoint generator from the starting coordinates (first 2 values) to the goal (last 2 values). "
                    f"The robot has two arms of lengths {str(arm_lengths[0])} and {str(arm_lengths[1])} and operates on an X,Y plane. "
                    f"Give the trajectory to follow in JSON format with the key 'waypoints'."
                    f"You can space out the waypoints"
                    )
        
        demonstration_filepath, demonstration_context_filepath = self._generate_prompt_file(self.selected_demonstrations, 
                            num_samples=10, 
                            context={
                                #  give overview as a string of arm lengths
                                "overview": overview,
                                "goal": self._convert_to_prompt_tokens(goal_position), 
                                "initial_position": self._convert_to_prompt_tokens(starting_position)})

        
        
        input_string = f" {self._convert_to_prompt_tokens(starting_position)} -> {self._convert_to_prompt_tokens(goal_position)}\n"
        generated_pathways = self.llm_client.process_ensemble_training(demonstration_context_filepath, n = self.number_of_ensomble_trajectories, temperature=self.temperture)
        
        # save the generated pathways
        with open("_generated_pathways.txt", 'w') as f:
            for pathway in generated_pathways:
                f.write(pathway + "\n")
                f.write(str(type(pathway)))

        generated_pathways_extracted = []
        for i in range(len(generated_pathways)):
                waypoints: List[List[int]] = self._extract_waypoints(str(generated_pathways[i]))["waypoints"]
                generated_pathways_extracted.append(waypoints)

        logprob_dict = self.llm_client.get_softmax_of_pathways(demonstration_filepath, generated_pathways_extracted, input=input_string)
        help, outputdictionary = self.llm_client.create_sub_set(logprob_dict, confidence_level=self.confidence_level)

    
        # conver the token trajectory to waypoints
        for i in range(len(outputdictionary["pathways"])):
            waypoints: list[list[int]] = self._extract_waypoints(str(outputdictionary["pathways"][i]["action_token_trajectory"]))["waypoints"]
            for j in range(len(waypoints)):
                waypoints[j] = [x / 100 for x in waypoints[j]]
            outputdictionary["pathways"][i]["waypoint_trajectory"] = waypoints

        return waypoints, outputdictionary

    def _extract_waypoints(self, raw_waypoints: str) -> dict:
        '''
        Extract the waypoints from the raw waypoints
        '''
        # convert string to a list of lists 
        # Use regular expressions to find all pairs of numbers

        # get rid of waypoint starting
        waypoints = re.findall(r'\[(-?\d+),\s*(-?\d+)\]', raw_waypoints)

        # Convert to list of lists with integers
        waypoints_list = [[int(x), int(y)] for x, y in waypoints]

        return_dict = {"waypoints": waypoints_list}
        return return_dict



    def calibrate(self):
        demonstrations = self.selected_demonstrations

        # initalise empty nonconformity scores
        non_conformity_scores = []
        correctly_guessed = 0
        correct_score_list = []
        # for each demonstration in demonstrations
        for i in range(len(demonstrations)):
            print("=============NEW CALIBRATION RUN [ ", str(i)," ]====================")
            # get subset of demonstrations that does not include the current demonstration
            sub_set_demonstrations = [d for d in demonstrations if d != demonstrations[i]]

            # get the input for current demonstration and divide by 100
            goal_position = demonstrations[i]['goal'] 
            # cretea a copy of the goal position
            goal_position = goal_position.copy()
            # divide  by 100
            for j in range(len(goal_position)):
                goal_position[j] = goal_position[j] / 100
            
            starting_position = demonstrations[i]['initial_position']
            # cretea a copy of the goal position
            starting_position = starting_position.copy()
            for j in range(len(starting_position)):
                starting_position[j] = starting_position[j] / 100


            correct_trajectory = demonstrations[i]['ee_states']
            
            
            overview = (
                    f"You are a waypoint generator from the starting coordinates (first 2 values) to the goal (last 2 values). "
                    f"The robot has two arms of lengths 1 and 1 and operates on an X,Y plane. "
                    f"Give the trajectory to follow in JSON format with the key 'waypoints'."
                    f"You can space out the waypoints"
                    )

            # generate prompt
            demonstration_filepath, demonstration_context_filepath = self._generate_prompt_file(sub_set_demonstrations, 
                            context={
                                #  give overview as a string of arm lengths
                                "overview": overview,
                                "goal": self._convert_to_prompt_tokens(goal_position), 
                                "initial_position": self._convert_to_prompt_tokens(starting_position)})

            # generate possible trajectories gets a list of possible trajectories
            possible_trajectories: List[str] = self.llm_client.process_ensemble_training(demonstration_context_filepath, n = self.number_of_ensomble_trajectories, temperature=self.temperture)
            # extract waypoints from the possible trajectories

            with open("_generated_pathways_calibration.txt", 'w') as f:
                for pathway in possible_trajectories:
                    f.write(pathway + "\n")
                    f.write(str(type(pathway)))
            
            for i in range(len(possible_trajectories)):
                waypoints: list[list[int]] = self._extract_waypoints(possible_trajectories[i])["waypoints"]
                possible_trajectories[i] = waypoints
            
            # append the correct trajectory 
            possible_trajectories.append(correct_trajectory)

            # shuffle trajectories
            np.random.shuffle(possible_trajectories)

            # get the softmax scores
            starting_position_tokens = self._convert_to_prompt_tokens(starting_position)
            goal_position_tokens = self._convert_to_prompt_tokens(goal_position)
            input_string = f" {starting_position_tokens} -> {goal_position_tokens}\n"
            logprob_dict = self.llm_client.get_softmax_of_pathways(demonstration_filepath, possible_trajectories,  input=input_string)

            # get the score for the correct trajectory
            # iterate through the keys of log prob
            max_incorrect_score = -1
            correct_score = -1
            for key in logprob_dict.keys():
                if logprob_dict[key]["trajectory"] == correct_trajectory:
                    correct_score = logprob_dict[key]["logprob"]
                    print("Correct Index: ", key)
                else:
                    if logprob_dict[key]["logprob"] > max_incorrect_score:
                        max_incorrect_score = logprob_dict[key]["logprob"]

            # calculate non conformity score; max(incorect) - correct
            non_conformity_score = max_incorrect_score - correct_score


            if correct_score > max_incorrect_score:
                print("Correctly guessed")
                correctly_guessed += 1
            else:
                print("Incorrectly guessed")

            print("Correct score: ", correct_score)

            # append non conformity score to list
            non_conformity_scores.append(non_conformity_score)
            correct_score_list.append(correct_score)

            time.sleep(10)

        # sort the non conformity scores
        correct_score_list.sort()

  
        non_conformity_score_threshold = int((len(correct_score_list) + 1)*(1 - self.delta))/len(correct_score_list)

        # linearly interpolate the threshold
        # confidence_level = np.percentile(non_conformity_scores, non_conformity_score_threshold)
        confidence_level = np.percentile(correct_score_list, self.delta)
        self.confidence_level = confidence_level

        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("====================================")
        print("Correctly guessed: ", correctly_guessed ,"/", len(demonstrations))
        print("conformity scores: ", correct_score_list)
        print("Threshold: ", confidence_level)
        print("====================================")


    def train(self, config: dict) -> dict:
        '''
        train the algorithm called with a different configuration each time

        returns:
        - a dictionary in the form of: 
        {
            threshold: float,
            help_needed: bool,
            pathways: [{
                action_token_trajectory: List[List[int,int]],
                waypoint_trajectory: List[List[int,int]],
                logprob: float,
                prediction_set: bool
            }] ... ,
        }
        '''

        # query llm
        waypoints, outputdictionary = self._query_llm(config['initial_position'], config['goal'], [config['l1'], config['l2']])

        # convert the token trajectory to waypoints
        return outputdictionary
        # raise NotImplementedError("Subclasses must implement train method")

    def test(self, config: dict):

        arm_lengths = [config['l1'], config['l2']]
        goal_position = config['goal']
        starting_position = config['initial_position']

        overview = (
                    f"You are a waypoint generator from the starting coordinates (first 2 values) to the goal (last 2 values). "
                    f"The robot has two arms of lengths {str(arm_lengths[0])} and {str(arm_lengths[1])} and operates on an X,Y plane. "
                    f"Give the trajectory to follow in JSON format with the key 'waypoints'."
                    f"You can space out the waypoints"
                    )

        demonstration_filepath, demonstration_context_filepath = self._generate_prompt_file(self.selected_demonstrations, 
                            num_samples=10, 
                            context={
                                "overview": overview,
                                "goal": self._convert_to_prompt_tokens(goal_position), 
                                "initial_position": self._convert_to_prompt_tokens(starting_position)})


        action_token_trajectory = self.llm_client.process_test(demonstration_context_filepath)
        waypoint_trajectory = self._extract_waypoints(action_token_trajectory)["waypoints"]

        for j in range(len(waypoint_trajectory)):
            waypoint_trajectory[j] = [x / 100 for x in waypoint_trajectory[j]]


        
        output_dictionary = {
            "action_token_trajectory": action_token_trajectory,
            "waypoint_trajectory": waypoint_trajectory}
        
        return output_dictionary