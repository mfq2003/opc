"""本模块集中定义点级 Recipe PPO 的稳定、无 GPU 依赖数据协议常量。

训练环境、共享 PPO、质量审核与后续标签转换必须从这里读取相同版本号，避免只读审核为了获得
常量而导入 Gymnasium、PyTorch 或 OpenILT。修改这些常量意味着产物协议变化，必须同步迁移数据。
"""

RECIPE_ENV_VERSION = "simpleopc-recipe-point-v1"
RECIPE_POINT_VERSION = "target-recipe-point-v1"
RECIPE_OBSERVATION_VERSION = "local-raster-64-v1"
PPO_RECIPE_LABEL_VERSION = "ppo-recipe-point-v1"
