from helios.configs.rl.maze.prior_initialized.base_conf import *
from helios.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from helios.data.maze.src.maze_agents import MazeActionPriorSACAgent

agent_config.update(AttrDict(
    td_schedule_params=AttrDict(p=1.),
))

agent_config.policy = ACLearnedPriorAugmentedPIPolicy
configuration.agent = MazeActionPriorSACAgent
