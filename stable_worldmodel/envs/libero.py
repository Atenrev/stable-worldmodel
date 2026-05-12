import gymnasium as gym
import numpy as np
from gymnasium import spaces

TASK_MAPPING = {
    0: ("libero_10", 4), 1: ("libero_10", 6), 2: ("libero_10", 9), 3: ("libero_10", 2),
    4: ("libero_10", 7), 5: ("libero_10", 0), 6: ("libero_10", 8), 7: ("libero_10", 1),
    8: ("libero_10", 3), 9: ("libero_10", 5), 10: ("libero_goal", 8), 11: ("libero_goal", 9),
    12: ("libero_goal", 3), 13: ("libero_goal", 6), 14: ("libero_goal", 2), 15: ("libero_goal", 5),
    16: ("libero_goal", 7), 17: ("libero_goal", 1), 18: ("libero_goal", 4), 19: ("libero_goal", 0),
    20: ("libero_object", 9), 21: ("libero_object", 4), 22: ("libero_object", 1), 23: ("libero_object", 3),
    24: ("libero_object", 0), 25: ("libero_object", 7), 26: ("libero_object", 2), 27: ("libero_object", 6),
    28: ("libero_object", 5), 29: ("libero_object", 8), 30: ("libero_spatial", 6), 31: ("libero_spatial", 4),
    32: ("libero_spatial", 5), 33: ("libero_spatial", 7), 34: ("libero_spatial", 0), 35: ("libero_spatial", 3),
    36: ("libero_spatial", 8), 37: ("libero_spatial", 1), 38: ("libero_spatial", 2), 39: ("libero_spatial", 9)
}

class Libero(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, max_episode_steps=600, render_mode="rgb_array", **kwargs):
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._vec_env = None
        self.task_index = None
        self.relative_idx = None
        
        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(0, 255, (256, 256, 3), dtype=np.uint8)
        })

    def _create_env(self, task_index):
        if self._vec_env is not None and self.task_index == task_index:
            return
            
        self.task_index = task_index
        suite, task_id = TASK_MAPPING.get(task_index, ("libero_10", 0))
        
        from lerobot.envs.libero import create_libero_envs
        import gymnasium as gym
        
        if self._vec_env is not None:
            self._vec_env.close()

        envs_dict = create_libero_envs(
            task=suite,
            n_envs=1,
            gym_kwargs={"task_ids": [task_id]},
            camera_name="agentview_image,robot0_eye_in_hand_image",
            episode_length=self.max_episode_steps,
            env_cls=gym.vector.SyncVectorEnv,
        )
        self._vec_env = envs_dict[suite][task_id]
        
        # update action space from the unwrapped single env
        single_env = self._vec_env.envs[0].unwrapped
        self.action_space = single_env.action_space
        
        # update observation_space to match _process_obs output
        sample_obs, _ = self._vec_env.reset()
        processed_obs = self._process_obs(sample_obs)
        space_dict = {}
        for k, v in processed_obs.items():
            if isinstance(v, np.ndarray):
                low = 0 if v.dtype == np.uint8 else -np.inf
                high = 255 if v.dtype == np.uint8 else np.inf
                space_dict[k] = spaces.Box(low=low, high=high, shape=v.shape, dtype=v.dtype)
            else:
                space_dict[k] = spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
        self.observation_space = spaces.Dict(space_dict)

    def set_task(self, task_index):
        if isinstance(task_index, (np.ndarray, list)):
            task_index = int(task_index[0])
        else:
            task_index = int(task_index)
        self._create_env(task_index)
        
    def set_relative_idx(self, relative_idx):
        if isinstance(relative_idx, (np.ndarray, list)):
            relative_idx = int(relative_idx[0])
        else:
            relative_idx = int(relative_idx)
        self.relative_idx = relative_idx
        if self._vec_env is not None:
            self._vec_env.envs[0].unwrapped.init_state_id = self.relative_idx
            # VERY IMPORTANT: resetting the environment after setting relative_idx
            self._vec_env.reset()
            
    def _set_state(self, state: np.ndarray):
        if self._vec_env is not None:
            single_env = self._vec_env.envs[0].unwrapped
            if hasattr(single_env, 'env'):
                rs_env = single_env.env
                rs_env.set_robot_joint_positions(state[:7])
                # step with 0 action to update physics/rendering
                try:
                    self._vec_env.step(np.zeros((1, 7), dtype=np.float32))
                except Exception as e:
                    print(f"Warning: Could not step Libero env after setting state: {e}")
                
    def _set_goal_state(self, goal_state: np.ndarray):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._vec_env is None:
            self._create_env(0)
            
        if self.relative_idx is not None:
            self._vec_env.envs[0].unwrapped.init_state_id = self.relative_idx
            
        obs, info = self._vec_env.reset(seed=seed, options=options)
        
        info_out = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray) and len(v) == 1:
                info_out[k] = v[0]
            else:
                info_out[k] = v
                
        processed_obs = self._process_obs(obs)
        
        # Ensure no overlap between observation keys and info keys
        overlapping_keys = list(set(processed_obs.keys()).intersection(set(info_out.keys())))
        for k in overlapping_keys:
            info_out.pop(k)

        # Force remove ANY key from info_out that matches processed_obs to be absolutely sure
        for k in list(processed_obs.keys()):
            if k in info_out:
                del info_out[k]

        return processed_obs, info_out
        
    def step(self, action):
        # Env is a single env wrapper, so action is (7,)
        # We need to pass (1, 7) to SyncVectorEnv
        obs, reward, terminated, truncated, info = self._vec_env.step(action[None])
        
        reward = float(reward[0])
        terminated = bool(terminated[0])
        truncated = bool(truncated[0])
        
        info_out = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray) and len(v) == 1:
                info_out[k] = v[0]
            else:
                info_out[k] = v
                
        if "is_success" in info_out:
            info_out["success"] = info_out["is_success"]
            
        processed_obs = self._process_obs(obs)
        for k in list(processed_obs.keys()):
            if k in info_out:
                del info_out[k]
            
        return processed_obs, reward, terminated, truncated, info_out
        
    def _process_obs(self, obs):
        if "pixels" in obs and isinstance(obs["pixels"], dict):
            obs["pixels"] = obs["pixels"]["image"]
        elif "image" in obs:
            obs["pixels"] = obs["image"]
        
        if hasattr(obs["pixels"], "shape"):
            if len(obs["pixels"].shape) == 4:
                obs["pixels"] = obs["pixels"][0]
            if obs["pixels"].shape[0] == 3:
                obs["pixels"] = obs["pixels"].transpose(1, 2, 0)
            
            if isinstance(obs["pixels"], np.ndarray):
                obs["pixels"] = np.ascontiguousarray(obs["pixels"])
                
                if obs["pixels"].dtype != np.uint8:
                    if obs["pixels"].max() <= 1.05:
                        obs["pixels"] = (obs["pixels"] * 255).astype(np.uint8)
                    else:
                        obs["pixels"] = obs["pixels"].astype(np.uint8)
        
        # Ensure obs only returns elements for single env, and exclude pixels
        obs_out = {}
        for k, v in obs.items():
            if k == "pixels" or k == "image":
                continue
            else:
                # If there are other states, remove the batch dim
                if isinstance(v, np.ndarray) and len(v.shape) > 0 and v.shape[0] == 1:
                    obs_out[k] = v[0]
                else:
                    obs_out[k] = v
        return obs_out

    def render(self):
        if hasattr(self._vec_env.envs[0].unwrapped, "render"):
            img = self._vec_env.envs[0].unwrapped.render()
            if img is not None:
                img = np.ascontiguousarray(img)
                return img
        return None
        
    def close(self):
        if self._vec_env is not None:
            self._vec_env.close()
