from helios.configs.hrl.kitchen.spirl.conf import *
from helios.models.closed_loop_helios_mdl import HELIOS_Prior_Mdl
from helios.rl.policies.cl_model_policies import ClModelPolicy
ll_model_params.cond_decode = True

# create LL closed-loop policy
ll_policy_params = AttrDict(
    policy_model=HELIOS_Prior_Mdl, 
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                         "skill_prior_learning/kitchen/helios_h_cl"
                                        # "test1/skill_prior_learning/kitchen/spirl_DPMM_h_cl"
                                        #  "skill_prior_learning/kitchen/spirl_DPMM_h_cl/gru_debug2"
                                        # "skill_prior_learning/kitchen/spirl_DPMM_h_cl/gru2sf001"
                                        # "skill_prior_learning/kitchen/spirl_DPMM_h_cl/gru2layers"
                                        # "skill_prior_learning/kitchen/spirl_DPMM_h_cl/gru1lsf001"
                                         ), #TODO: change pretrained weights dir according to your 
)
ll_policy_params.update(ll_model_params)

# create LL SAC agent (by default we will only use it for rolling out decoded skills, not finetuning skill decoder)
ll_agent_config = AttrDict(
    policy=ClModelPolicy,
    policy_params=ll_policy_params,
    critic=MLPCritic,               
    critic_params=hl_critic_params
)

# update HL policy model params
hl_policy_params.update(AttrDict(
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
))

# register new LL agent in agent_config and turn off LL agent updates
agent_config.update(AttrDict(
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    update_ll=False,
))


