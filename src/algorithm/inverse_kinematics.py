import numpy as np

def inverse_kinematics(env, current_state):
    goal_theta1, goal_theta2 = env.inverse_kinematics(*env.goal)

    # Calculate the actions needed to move towards the goal joint angles
    action1 = goal_theta1 - env.theta1
    action2 = goal_theta2 - env.theta2

    # Normalize the actions to the range [-π, π]
    action1 = (action1 + np.pi) % (2 * np.pi) - np.pi
    action2 = (action2 + np.pi) % (2 * np.pi) - np.pi

    # Clip the actions to ensure they do not exceed the action space limits
    action1 = np.clip(action1, -env.action_max_min, env.action_max_min)
    action2 = np.clip(action2, -env.action_max_min, env.action_max_min)

    return np.array([action1, action2])
