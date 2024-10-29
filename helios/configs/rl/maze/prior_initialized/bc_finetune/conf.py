from helios.configs.rl.maze.prior_initialized.base_conf import *
from helios.rl.policies.prior_policies import ACPriorInitializedPolicy
from helios.data.maze.src.maze_agents import MazeSACAgent

agent_config.policy = ACPriorInitializedPolicy
configuration.agent = MazeSACAgent
