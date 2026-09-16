"""本模块把无框架的 v2 LocalEPEEpisode 适配为 Gymnasium 单点离散动作环境。

输入是已经绑定固定 FRAG、独立 Golden evaluator 和冻结 observation 的 LocalEPEEpisode；输出是
Gymnasium Dict observation、Discrete action 与 dense/terminal reward。这个类只允许普通 SB3 PPO 的
全动作合法配置；masked policy 尚未实现独立 wrapper/trainer，不能通过一个布尔开关绕过构造期门槛。
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .recipe_v2 import LocalEPEEpisode
from .recipe_v2_contract import ACTOR_GEOMETRY_FIELDS


class LocalEPEPPOEnv(gym.Env):
    """逐点接收离散法向动作，并把 v2 核心协议暴露为标准 Gymnasium API。"""

    metadata = {"render_modes": []}

    def __init__(self, episode: LocalEPEEpisode):
        if not isinstance(episode, LocalEPEEpisode):
            raise TypeError("episode 必须是 LocalEPEEpisode")
        # 在 Gym/PPO 构造期、baseline solver 调用前硬失败；不能依赖调用方记得手动 preflight。
        episode.require_plain_ppo_compatible()
        super().__init__()
        self.episode = episode
        patch_size = int(episode.patch_size)
        self.action_space = spaces.Discrete(len(episode.action_offsets_nm))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, patch_size, patch_size),
                dtype=np.float32,
            ),
            "vector": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(ACTOR_GEOMETRY_FIELDS),),
                dtype=np.float32,
            ),
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """恢复全零 Recipe；可通过 options.point_order 注入完整确定性 schedule。"""
        super().reset(seed=seed)
        point_order = None if options is None else options.get("point_order")
        observation, info = self.episode.reset(seed=seed, point_order=point_order)
        return observation, info

    def step(self, action):
        """执行当前 EPE 点动作；v2 核心负责完整性、reward 和 solver-call 约束。"""
        if not self.action_space.contains(action):
            raise ValueError("action 必须由当前 Discrete action_space 接受")
        observation, reward, terminated, info = self.episode.step(action)
        return observation, float(reward), bool(terminated), False, info

    def action_masks(self) -> np.ndarray:
        """返回诊断掩码；本普通 PPO wrapper 已要求其全为真，不代表已实现 masked trainer。"""
        return self.episode.action_mask()

    def assert_plain_ppo_compatible(self) -> None:
        """普通 SB3 PPO 训练前拒绝任何包含非法动作的几何配置。"""
        self.episode.require_plain_ppo_compatible()
