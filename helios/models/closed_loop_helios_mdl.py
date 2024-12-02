import torch
import bnpy
import os
from itertools import cycle
import numpy as np

from helios.utils.general_utils import batch_apply
from helios.utils.pytorch_utils import get_constant_parameter
from helios.models.skill_prior_mdl import SkillPriorMdl
from helios.modules.subnetworks import Predictor, BaseProcessingLSTM
from helios.modules.recurrent_modules import BaseProcessingGRU
from helios.modules.variational_inference import MultivariateGaussian
from helios.modules.losses import KLDivLoss, NLL, DPM_KLDLoss
from helios.utils.general_utils import AttrDict
from helios.modules.variational_inference import Gaussian, MultivariateGaussian, get_fixed_prior
from helios.components.checkpointer import load_by_key, freeze_modules

from bnpy.data.XData import XData # pip install bnpy


class HELIOS_Prior_Mdl(SkillPriorMdl):
    """HELIOS model with closed-loop low-level skill decoder.
        Basic implementation based on: https://github.com/Ghiara/DIVA/blob/master/diva.py
    """

    def __init__(self, params, logger=None):
        super().__init__(params, logger=logger)
        pwd = os.getcwd()
        self.dpmm_folder_name = params.dpmm_folder_name if hasattr(params, 'dpmm_folder_name') else 'default'
        self.bnp_root = pwd + f'/save/bn_model/{self.dpmm_folder_name}/'
        self.bnp_iterator = cycle(range(2))
        self.bnp_model = None
        self.bnp_info_dict = None
        self.comp_var = None
        self.comp_mu = None
        self.num_clusters = 0
        self.cluster_logging = []
        self.dpmm_param = dict(
            # sF=1.0 # debug1
            # sF=0.1, # debug2
            # sF=0.001, # debug3
            
            ## -- mnist default setup --
            # b_minNumAtomsForNewComp=16.0,
            # b_minNumAtomsForTargetComp=16.0,
            # b_minNumAtomsForRetainComp=16.0,
            ## debug 4
            # b_minNumAtomsForNewComp=2000.0,
            # b_minNumAtomsForTargetComp=2000.0,
            # b_minNumAtomsForRetainComp=2000.0,
            ## debug 5
            # b_minNumAtomsForNewComp=500.0,
            # b_minNumAtomsForTargetComp=600.0,
            # b_minNumAtomsForRetainComp=600.0,
            
            sF=0.01,
            b_minNumAtomsForNewComp=800.0,
            b_minNumAtomsForTargetComp=850.0,
            b_minNumAtomsForRetainComp=850.0,

        ) 

    def build_network(self, inference_block_type:str='lstm'):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net(inference_block_type)
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0., learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self, block_type:str='lstm'):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        if block_type in ['lstm']:
            return torch.nn.Sequential(
                BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
                torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
            )
        # -- implementation of gru network --
        elif block_type in ['gru']:
            return torch.nn.Sequential(
            BaseProcessingGRU(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),  # GRU-based layer
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)  # Output layer for VAE
        )
        else:
            raise NotImplementedError


    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1]) 
        # data = self.q(inf_input)[:, -1]
        # return MultivariateGaussian(data) 


    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the SPIRL_DPMM model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions    # for seamless evaluation

        # run inference
        output.q = self._run_inference(inputs) # q(z|a)

        # compute (fixed) prior
        output.p = get_fixed_prior(output.q) # p(z) ~ N(0,1)

        # infer learned skill prior, p(z|s0)
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        if use_learned_prior:
            output.p = output.q_hat     # use output of learned skill prior for sampling

        # -- sample DPMM variable --
        if self._sample_prior:
            
            if not use_learned_prior and self.bnp_model:
                # Validation based on DPMM 
                z = output.q.sample()
                _, hard_assignment = self.cluster_assignments(z)  # [batch_size]
                zs_sampled = []
                for i in range(len(hard_assignment)):
                    k = hard_assignment[i]
                    z_sampled = torch.distributions.MultivariateNormal(
                        loc=self.comp_mu[k].to(z.device),
                        covariance_matrix=torch.diag_embed(self.comp_var[k],
                    ).to(z.device)).sample()
                    zs_sampled.append(z_sampled)
                z_component = torch.stack(zs_sampled, dim=0)
                output.z = z_component
                output.z_q = z # for loss computation
            else:
                # Validation based on Gauss 
                output.z = output.p.sample()
                output.z_q = output.q.sample() # for loss computation
        else: # if training/inference
            output.z = output.q.sample()
            output.z_q = output.z.clone() # for loss computation

        # decode
        assert self._regression_targets(inputs).shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        return output
    
    def loss(self, model_output, inputs):
        """Loss computation of the SPIRL_DPMM model.
        :arg model_output: output of SPIRL_DPMM model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """

        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian

        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight) \
            (Gaussian(model_output.reconstruction, torch.zeros_like(model_output.reconstruction)),
             self._regression_targets(inputs))

        if self.bnp_model is None:
            losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)
        else: 
            z = model_output.z_q.detach()          
            comp_mu = self.comp_mu
            comp_var = self.comp_var
            prob_comps, dpm_hard_assignment = self.cluster_assignments(z) # prob_comps --> resp, comps --> Z[n]
            _, self.num_clusters = prob_comps.shape
            losses.kl_loss = DPM_KLDLoss(self.beta)(model_output.q.mu, model_output.q.log_sigma, prob_comps, comp_mu, comp_var)

        # learned skill prior net loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_bedta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)

        return losses
    
    def fit_dpmm(self, z):
        """
        Update DPM model at the end of the epoch. Implementation inherits from https://github.com/Ghiara/DIVA/blob/master/diva.py
        :arg z: latent embeddings
        """
        z = XData(z.detach().cpu().numpy())

        
        if not self.bnp_model:
          print("-- Initialing DPMM model --")
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                        output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                        initname='randexamples',
                                                        K=1, 
                                                        gamma0 = 5.0, 
                                                        sF=0.1, 
                                                        ECovMat='eye',
                                                        b_Kfresh=5, b_startLap=0, m_startLap=2,
                                                        # moves='birth,merge,shuffle', 
                                                        moves='birth,delete,merge,shuffle', 
                                                        nLap=2,
                                                        b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'],
                                                        b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'],
                                                        b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'],
                                                        )
        
        else:
          print("-- Fitting DPMM model --") 
          self.bnp_model, self.bnp_info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB', 
                                                        output_path = self.bnp_root+str(next(self.bnp_iterator)),
                                                        initname=self.bnp_info_dict['task_output_path'],
                                                        K=self.bnp_info_dict['K_history'][-1], # inherits from last update
                                                        gamma0=5.0,
                                                        sF=self.dpmm_param['sF'],
                                                        ECovMat='eye',
                                                        b_Kfresh=5, b_startLap=1, m_startLap=2,
                                                        # moves='birth,merge,shuffle', 
                                                        moves='birth,delete,merge,shuffle', 
                                                        nLap=2,
                                                        b_minNumAtomsForNewComp=self.dpmm_param['b_minNumAtomsForNewComp'],
                                                        b_minNumAtomsForTargetComp=self.dpmm_param['b_minNumAtomsForTargetComp'],
                                                        b_minNumAtomsForRetainComp=self.dpmm_param['b_minNumAtomsForRetainComp'],
                                                        )
        self.calc_cluster_component_params()

    
    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.bnp_model.calc_local_params(z)
        # Here, resp is a 2D array of size N x K. here N is batch size, K active clusters
        # Each entry resp[n, k] gives the probability that data atom n is assigned to cluster k under 
        # the posterior.
        resp = LP['resp'] 
        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        # Z represents for each atom n (in total N), which cluster it should belongs to accroding to the probability
        Z = resp.argmax(axis=1)
        return resp, Z

    def calc_cluster_component_params(self):
        self.comp_mu = [torch.Tensor(self.bnp_model.obsModel.get_mean_for_comp(i)) for i in np.arange(0, self.bnp_model.obsModel.K)]
        self.comp_var = [torch.Tensor(np.sum(self.bnp_model.obsModel.get_covar_mat_for_comp(i), axis=0)) for i in np.arange(0, self.bnp_model.obsModel.K)]  
    
    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])
        else:
            super().load_weights_and_freeze()

    @property
    def enc_size(self):
        return self._hp.state_dim



