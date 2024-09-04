import sys
import importlib
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent
sys.path.append(str(src_path))

from env.simple_robot_arm_env import SimpleRobotArmEnv
from utils.file_io import load_config, save_data, create_folder_if_not_exists

def main():
    # Load configuration
    config = load_config(str(src_path / 'config/config.json'))

    # Create the environment with the loaded configuration
    env = SimpleRobotArmEnv(config)

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

        print(f"Action: {action}, State: {ee_state}, Reward: {reward}")

    # Create output folder if it doesn't exist
    create_folder_if_not_exists(config['output_folder'])

    # Save the collected data
    save_data({
        "config": config,
        "data": {
            "joint_states": env.joint_states,
            "ee_states": env.ee_states,
            "rewards": env.rewards,
            "dones": env.dones,
            "actions": env.actions
        }
    }, config['output_folder'] + '/output.pkl')

    # Render the collected actions to a video file
    env.render_to_video(config['output_folder'] + '/robot_arm_animation.mp4')

    # Replay the demonstration from the saved data
    env.replay_demonstration(config['output_folder'] + '/output.pkl')

if __name__ == '__main__':
    main()
