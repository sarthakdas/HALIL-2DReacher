import numpy as np

def another_algorithm(env, current_state):
    # An example of another algorithm that simply tries to move in small steps towards the goal
    action = np.array([0.01, 0.01])
    return np.clip(action, -env.action_max_min, env.action_max_min)
