import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

GOAL_THRESHOLD = 0.1

class SimpleRobotArmEnv:
    def __init__(self, config):
        self.action_max_min = config['action_max_min']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.goal = np.array(config['goal'])
        self.initial_position = config['initial_position']
        self.max_steps = config['max_steps']
        self.output_folder = config['output_folder']
        self.algorithm_name = config['algorithm']
        self.reset()
        self.all_waypoints = []

        self.current_algorithm_name = None

        print("Envrioment Initialised to: ", self.initial_position, "->", self.goal)

    # def inverse_kinematics(self, x, y):
    #     cos_angle2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
    #     cos_angle2 = np.clip(cos_angle2, -1, 1)  # Clip cos_angle2 to the valid range [-1, 1]
        
    #     sin_angle2 = np.sqrt(1 - cos_angle2**2)
    #     theta2 = np.arctan2(sin_angle2, cos_angle2)
        
    #     k1 = self.l1 + self.l2 * cos_angle2
    #     k2 = self.l2 * sin_angle2
        
    #     theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
    #     return theta1, theta2
    
    # def forward_kinematics(self, theta1, theta2):
    #     x1 = self.l1 * np.cos(theta1)
    #     y1 = self.l1 * np.sin(theta1)
    #     x2 = x1 + self.l2 * np.cos(theta1 + theta2)
    #     y2 = y1 + self.l2 * np.sin(theta1 + theta2)
    #     return (0, 0), (x1, y1), (x2, y2)

    def inverse_kinematics(self, x, y):
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

    def forward_kinematics(self, theta1, theta2):
        # Normalize angles to the range [-π, π]
        theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi

        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        x2 = x1 + self.l2 * np.cos(theta1 + theta2)
        y2 = y1 + self.l2 * np.sin(theta1 + theta2)
        
        return (0, 0), (x1, y1), (x2, y2)
    
    def reset(self):
        self.theta1, self.theta2 = self.inverse_kinematics(*self.initial_position)
        return np.array(self.forward_kinematics(self.theta1, self.theta2)[-1])
    
    def step(self, action):
        # Clip the action to the valid range
        action[0] = np.clip(action[0], -self.action_max_min, self.action_max_min)
        action[1] = np.clip(action[1], -self.action_max_min, self.action_max_min)
        self.theta1 += action[0]
        self.theta2 += action[1]
        _, _, (x, y) = self.forward_kinematics(self.theta1, self.theta2)
        ee_state = np.array([x, y])
        distance_to_goal = np.linalg.norm(ee_state - self.goal)
        reward = -distance_to_goal
        done = distance_to_goal < GOAL_THRESHOLD
        joint_position_state = np.array([self.theta1, self.theta2])
        return ee_state, joint_position_state, reward, done
    
    
    def render_to_video(self, filename='robot_arm_animation.mp4', states=None):
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # Adjust the figure to make room for the debug text

        line1, = ax.plot([], [], 'ro-', lw=2)
        line2, = ax.plot([], [], 'bo-', lw=2)
        goal_point, = ax.plot(self.goal[0], self.goal[1], 'gx')

        # Plotting waypoints
        if hasattr(self, 'all_waypoints') and self.all_waypoints is not None:
            waypoints_x = [wp[0] for wp in self.all_waypoints]
            waypoints_y = [wp[1] for wp in self.all_waypoints]
            ax.scatter(waypoints_x, waypoints_y, c='magenta', marker='x', label='Waypoints')
            ax.legend()

        if hasattr(self, 'ensomble_dictionary') and self.ensomble_dictionary is not None:
            for i in range(len(self.ensomble_dictionary["pathways"])):
                waypoints = self.ensomble_dictionary['pathways'][i]['waypoint_trajectory']
                waypoints_x = [wp[0] for wp in waypoints]
                waypoints_y = [wp[1] for wp in waypoints]

                marker_size = self.ensomble_dictionary['pathways'][i]['logprob'] * 20

                if self.ensomble_dictionary['pathways'][i]['prediction_set']:
                    color = 'red'
                else:
                    color = 'cyan'

                label = round(self.ensomble_dictionary['pathways'][i]['logprob'], 3)

                ax.scatter(waypoints_x, waypoints_y, c=color, marker='o', label=label, s=marker_size)
                ax.legend()

        # add title
        title_part_one = "Success" if self.dones[-1] else "Failure"
        title_part_two = self.current_algorithm_name
        ax.set_title(f"{title_part_one} - {title_part_two}")

    
        debug_text = ax.text(0.8, 0.9, '', transform=ax.transAxes)
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2, goal_point

        def animate(i):
            joint_position_state = states[i] if states is not None else self.joint_states[i]
            theta1, theta2 = joint_position_state
            _, (x1, y1), (x2, y2) = self.forward_kinematics(theta1, theta2)
            line1.set_data([0, x1], [0, y1])
            line2.set_data([x1, x2], [y1, y2])
            return line1, line2, goal_point, debug_text

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(states) if states is not None else len(self.joint_states), blit=True, interval=25)
        ani.save(filename, writer='ffmpeg')
        plt.close(fig)

    
    def save_demo(self, filename):
        output = {
            'joint_states': self.joint_states,
            'ee_states': self.ee_states,
            'rewards': self.rewards,
            'dones': self.dones,
            'actions': self.actions
        }
        with open(filename, 'wb') as f:
            pickle.dump(output, f)
    
    def replay_demonstration(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.render_to_video(self.output_folder + '/replay.mp4', states=data['data']['joint_states'])
