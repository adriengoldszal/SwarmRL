from gymnasium.spaces.space import Space
import numpy as np
from numpy import ndarray
import pygame
import gymnasium as gym
from gymnasium import spaces
import cv2
from swarm_env.env_renderer import GuiSR
from swarm_env.multi_env.ma_drone import MultiAgentDrone
import gc
from custom_maps.intermediate01 import MyMapIntermediate01
from custom_maps.easy import EasyMap
from typing import Any, Dict, Generic, Iterable, Iterator, TypeVar
from pettingzoo import ParallelEnv
from gymnasium.utils import EzPickle, seeding
from swarm_env.constants import *
import arcade

"""
Environment for multi agent
"""


map_dict = {
    "MyMapIntermediate01": MyMapIntermediate01,
    "Easy": EasyMap,
}


class MASwarmMarket(gym.Env):
    """
    Oservation
    GPS Position: 2
    Velocity: 2
    Humans: (distance, angle, grasped)
    Rescue zone: (distance, angle, grasped)
    Lidar: 180


    Reward
    - Every step: -0.05
    - If hit the wall: -5
    - If touch the person: +50
    - Entropy score # may not be neccesary as we have the delta exploration score
    - Exploration increase score

    Terminate when reach the wounded person

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_name="Easy",
        n_agents=1,
        n_targets=1,
        map_size=300,
        render_mode="rgb_array",
        max_episode_steps=100,
        continuous_action=True,
        fixed_step=20,
        share_reward=True,
        use_exp_map=False,
        use_conflict_reward=False,
    ):
        EzPickle.__init__(
            self,
            map_name=map_name,
            n_agents=n_agents,
            n_targets=n_targets,
            map_size=map_size,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            continuous_action=continuous_action,
            fixed_step=fixed_step,
        )
        self.map_side_size = map_size
        
        if map_name in map_dict:
            self.map_name = map_name
            self._map = map_dict[map_name](num_drones=n_agents, num_persons=n_targets,size=map_size)
        else:
            raise Exception("Invalid map name")

        self.map_size = self._map._size_area
        self.continuous_action = continuous_action
        self.share_reward = share_reward
        self.n_agents = n_agents
        self.n_targets = n_targets

        self._playground = None
        self._agents = None
        self._agents = None
        self.agents = [i for i in range(n_agents)]
        self.fixed_step = fixed_step
        self.persons = None
        self.use_exp_map = use_exp_map
        self.use_conflict_reward = use_conflict_reward
        
        self.previous_order = None

        ### OBSERVATION

        """
        Lidar: 180 + semantic: (1 + 3 + 2) * 3 + pose: 3 + velocity : 2 = 203
        """
        single_observation_dim = (
            180
            + (self.n_targets + self.n_agents) * 3
            + 5
            + (self.n_agents * self.n_targets)
            + 1  # encoding for the message from other drones
        )
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(single_observation_dim,))
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(single_observation_dim * self.n_agents,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]

        # forward, lateral, rotation, grasper and bids for each target
        self.action_space = [
            spaces.Box(
                low=np.array([-1, -1, -1, 0] + [0] * self.n_targets),
                high=np.array([1, 1, 1, 1] + [1] * self.n_targets),
                shape=(4 + self.n_targets,),
            )
            for agent_id in self.agents
        ]

        self.current_rescue_count = 0
        self.current_step = 0
        self.max_episode_steps = max_episode_steps
        self.last_exp_score = None
        self.render_mode = render_mode
        self.gui = None
        self.clock = None
        self.ep_count = 0

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def flatten_obs(self, observation):
        obs_array = np.array([])
        for v in observation.values():
            obs_array = np.concatenate((obs_array, v.flatten()), axis=0)
        return obs_array

    def state(self) -> ndarray:
        observations = self._get_obs()
        state_array = []
        for agent_id in self.agents:
            obs_array = self.flatten_obs(observations[agent_id])
            state_array.append(obs_array)
        return np.array(state_array)

    def construct_action(self, action):
        return {
            "forward": np.clip(action[0], -1, 1),
            "lateral": np.clip(action[1], -1, 1),
            "rotation": np.clip(action[2], -1, 1),
            "grasper": 1 if action[3] > 0.5 else 0,
        }, action[4:]

    def observe(self, agent_id):
        agent = self._agents[agent_id]
        lidar = agent.lidar_values()[:-1].astype(np.float32) / LIDAR_MAX_RANGE
        velocity = agent.measured_velocity().astype(np.float32)
        normalized_position = (
            agent.true_position()[0] / self.map_size[0],
            agent.true_position()[1] / self.map_size[1],
        )
        pose = np.concatenate(
            (normalized_position, [agent.true_angle()]), axis=0
        ).astype(np.float32)
        semantic = np.zeros((1 + self.n_targets + (self.n_agents - 1), 3)).astype(
            np.float32
        )
        center, human, drone = agent.process_special_semantic()

        semantic[0] = center[0]
        for i in range(min(len(human), self.n_targets)):
            semantic[1 + i] = human[i]

        for i in range(min(len(drone), self.n_agents - 1)):
            semantic[1 + self.n_targets + i] = drone[i]

        messages = np.zeros((self.n_agents, self.n_targets))
        for i, agent in enumerate(self._agents):
            messages[i] = agent.state["message"]

        grasper = [1] if len(self._agents[agent_id].grasped_entities()) > 0 else [0]

        observation = np.concatenate(
            [
                lidar,
                velocity,
                pose.flatten(),
                semantic.flatten(),
                grasper,
                messages.flatten(),
            ],
            axis=0,
        ).astype(np.float32)

        return observation

    def _get_obs(self):
        observations = []
        for name in self.agents:
            observation = self.observe(agent_id=name)
            observations.append(observation)
        return observations

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def get_agent_info(self, agent_id):
        info = {}
        info["drones_true_pos"] = {agent_id: self._agents[agent_id].true_position()}
        return info

    def _get_info(self):
        infos = {}
        infos["map_name"] = self.map_name
        infos["wounded_people_pos"] = self._map._wounded_persons_pos
        infos["rescue_zone"] = self._map._rescue_center_pos
        for agent_id in self.agents:
            infos[agent_id] = self.get_agent_info(agent_id)
        return infos

    def reset_map(self):
        self._map.explored_map.reset()
        self._map.reset_rescue_center()
        self._map.reset_wounded_person()
        self._map.reset_drone()

    def re_init(self):
        self._map = map_dict[self.map_name](
            num_drones=self.n_agents, num_persons=self.n_targets, size=self.map_side_size,
        )
        self.map_size = self._map._size_area
        self._playground = self._map.construct_playground(drone_type=MultiAgentDrone)
        self._agents = self._map.drones
        self.gui = GuiSR(self._playground, self._map)

    def reset(self, seed=None, options=None):
        if (
            self.ep_count % 1 == 0
        ):  # change to self.ep_count % 1 == 0 to avoid mem leak, but hurt performance
            del self._map
            del self._agents
            del self._playground
            del self.gui
            arcade.close_window()
            self.re_init()
        self.ep_count += 1
        self._playground.window.switch_to()
        self.reset_map()
        self._playground.reset()
        for agent in self._agents:
            agent.state["message"] = np.zeros((self.n_targets,))
        self.current_step = 0
        self.current_rescue_count = 0
        observation = self._get_obs()
        # info = self._get_info()
        return observation

    def render(self, mode=None):
        if mode:
            self.render_mode = mode
        return self._render_frame()

    def get_map(self):
        return self._map

    def process_order(self):
        """
        Process bid wins and avoid two winning bids by the same drone
        """
        order = [-1] * self.n_targets  # Initialize with -1 (unassigned)
        
        # Get all bids in a more manageable format
        # List of lists: [[drone0_bids], [drone1_bids], ...]
        all_bids = [[a.state["message"][i] for i in range(self.n_targets)] for a in self._agents]
        
        # Keep track of which drones are already assigned
        assigned_drones = set()
        
        # While there are unassigned targets
        while -1 in order:
            highest_bid = float('-inf')
            best_drone = -1
            best_target = -1
            
            # Find highest bid among unassigned targets and available drones
            for target_idx in range(self.n_targets):
                if order[target_idx] == -1:  # If target is unassigned
                    for drone_idx in range(len(self._agents)):
                        if drone_idx not in assigned_drones:  # If drone is available
                            bid = all_bids[drone_idx][target_idx]
                            if bid > highest_bid:
                                highest_bid = bid
                                best_drone = drone_idx
                                best_target = target_idx
            
            # Assign the best drone to the target
            if best_drone != -1:
                order[best_target] = best_drone
                assigned_drones.add(best_drone)
        
        print(f"Order {order}")
        return order

    def reward(self, idx, action):
        agent = self._agents[idx]
        rew = -np.abs(action[2])
        conflict = 0
        if agent.is_collided():
            rew -= 1
        if agent.touch_human():
            rew += 1
        for human in self._map._wounded_persons:
            magnets = set(human.grasped_by)
            if len(magnets) > 1 and agent.base.grasper in magnets:
                if self.use_conflict_reward:
                    rew -= 1
                conflict += 1

        ### Reward to instruct drone to respect the order
        current_order = self.process_order()
        
        if self.previous_order is not None:
            for i in range(len(current_order)):
                # If this target was previously assigned to this drone but isn't anymore
                if self.previous_order[i] == idx and current_order[i] != idx:
                    # Check if drone was actually grasping when it changed assignment
                    if agent.base.grasper in self._map._wounded_persons[i].grasped_by:
                        rew -= 0.75  # Higher penalty for releasing
                    else:
                        rew -= 0.25
                        
        for i in range(len(current_order)):
            if current_order[i] == idx:
                if agent.base.grasper in self._map._wounded_persons[i].grasped_by:
                    rew += 2
                else:
                    rew -= 0.25
                    
        self.previous_order = current_order
        
        return rew, conflict

    def step(self, actions):
        self._playground.window.switch_to()
        frame_skip = 5
        counter = 0
        done = False
        steps = self.fixed_step
        prev_distances = [0] * self.n_agents
        for i, person in enumerate(self._map._wounded_persons):
            position = person.position
            prev_distances[i] = self.get_distance(
                (position[0], position[1]),
                self._map._rescue_center_pos[0],
            )

        commands = {}
        for i, agent in enumerate(self._agents):
            move, msg = self.construct_action(actions[i])
            print(f"Message {msg} from agent {i}")
            agent.state["message"] = msg
            commands[agent] = move

        terminated, truncated = False, False
        rewards = [-0.5 for _ in range(self.n_agents)]

        while counter < steps and not done:
            _, _, _, done = self._playground.step(commands)

            for i, agent in enumerate(self._agents):
                if agent.reward != 0:
                    self.current_rescue_count += agent.reward
                    rewards[i] += 50

            if self.current_rescue_count >= self._map._number_wounded_persons:
                terminated = True
                self.current_rescue_count = 0
                break

            if self.render_mode == "human" and counter % frame_skip == 0:
                # self._agent.update_grid()
                self._render_frame()
            counter += 1

        conflicts = [0] * self.n_agents
        for i, agent in enumerate(self._agents):
            reward, conflict = self.reward(i, actions[i])
            rewards[i] += reward
            conflicts[i] += conflict
            
        print(f"Rewards before truncation penalty {rewards}")
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
            for i in range(len(rewards)):
                rewards[i] -= 20

        print(f"Rewards after truncation penalty {rewards}")
        
        # SHARED REWARD DEFINITION
        shared_reward = sum(rewards)
        print(f"Base shared reward {shared_reward}")
        if not truncated:
            shared_reward = max(shared_reward, -10 * self.n_agents)

        delta_distances = 0
        for i, person in enumerate(self._map._wounded_persons):
            position = person.position
            delta_distances += (
                self.get_distance(
                    (position[0], position[1]),
                    self._map._rescue_center_pos[0],
                )
                - prev_distances[i]
            )
        
        shared_reward -= delta_distances / 5
        print(f"delta_distances {delta_distances}, shared reward {shared_reward}")
        
        if self.use_exp_map:
            current_exp_score = self._map.explored_map.score()
            if self.last_exp_score is not None:
                delta_exp_score = current_exp_score - self.last_exp_score
            else:
                delta_exp_score = 0

            self.last_exp_score = current_exp_score
            # print(f"score {delta_exp_score}, {current_exp_score}")
            self.gui.update_explore_map()

            # REWARD
            shared_reward += 50 * delta_exp_score
            print(f"Add delta_exp ({50*delta_exp_score})")
        if self.share_reward:
            print(f"Final shared reward {shared_reward}")
            final_rewards = [[shared_reward]] * self.n_agents
        else:
            final_rewards = rewards
        # terminations = [terminated] * self.n_agents
        # truncations = [truncated] * self.n_agents
        dones = [terminated or truncated] * self.n_agents

        observations = self._get_obs()
        infos = self._get_info()

        infos["conflict_count"] = conflicts

        if self.render_mode == "human":
            self._render_frame()

        return observations, final_rewards, dones, infos

    def _render_frame(self):
        # Capture the frame
        image = self.gui.get_playground_image()

        if self.render_mode == "human":
            for name in self.agents:
                color = (255, 0, 0)
                offset = 10
                agent = self._agents[name]
                pt1 = (
                    agent.true_position()
                    + np.array(self.map_size) / 2
                    + np.array([offset, offset])
                )
                org = (int(pt1[0]), self.map_size[1] - int(pt1[1]))
                str_id = str(name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.putText(
                    image,
                    str_id,
                    org,
                    fontFace=font,
                    fontScale=0.4,
                    color=color,
                    thickness=1,
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()
            cv2.imshow("Playground Image", image)
            cv2.waitKey(1)
            self.clock.tick(self.metadata["render_fps"])

        return image

    def sample_action(self):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.action_space[i].sample())
        return actions

    def close(self):
        gc.collect()
        cv2.destroyAllWindows()

    def observation_space(self, agent: Any) -> Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> Space:
        return self.action_spaces[agent]
