from helios.configs.rl.kitchen.prior_initialized.base_conf import *
from helios.rl.policies.prior_policies import PriorInitializedPolicy

agent_config.policy = PriorInitializedPolicy
configuration.agent = SACAgent

