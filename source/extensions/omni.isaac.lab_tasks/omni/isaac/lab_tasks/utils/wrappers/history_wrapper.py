#! python3
# -*- encoding: utf-8 -*-
"""
@File    :   history_wrapper.py
@Time    :   2024/09/26 22:34:52
@Author  :   Cao Zhanxiang 
@Version :   1.0
@Contact :   caozx1110@163.com
@License :   (C)Copyright 2023
@Desc    :   None
"""

import gymnasium as gym
import torch

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env: ManagerBasedRLEnv, history_length: int = 20):
        super(HistoryWrapper, self).__init__(env)

        self.env: ManagerBasedRLEnv = env    
        self.history_length = history_length
        self.history = torch.zeros((self.unwrapped.scene.num_envs, self.history_length, *self.unwrapped.observation_manager.group_obs_dim["policy"]), device=self.unwrapped.device, dtype=torch.float32, requires_grad=False)
        
    def reset(self, **kwargs):
        obs_dict, extras = self.env.reset(**kwargs)
        self.history = torch.zeros((self.unwrapped.scene.num_envs, self.history_length, *self.unwrapped.observation_manager.group_obs_dim["policy"]), device=self.unwrapped.device, dtype=torch.float32, requires_grad=False)
        self.history[:, -1] = obs_dict['policy']
        # TEMP: history as the policy observation
        obs_dict['policy'] = self._get_policy_obs()
        
        return obs_dict, extras
    
    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)        
        
        # zero the history for the reset environments
        reset_env_ids = self.unwrapped.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.history[reset_env_ids] = 0
        
        # update the history
        self.history = torch.cat([self.history[:, 1:], obs_dict['policy'].unsqueeze(1)], dim=1)
        
        # TEMP: history as the policy observation
        obs_dict['policy'] = self._get_policy_obs()
        
        return obs_dict, rew, terminated, truncated, extras
            
    """
    Properties -- Gym.Wrapper
    """
        
    def _get_policy_obs(self):
        # return the history as the observation [num_envs, history_length * obs_shape]
        return self.history.flatten(start_dim=1, end_dim=2)
    
    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped
        
        

if __name__ == '__main__':
    pass