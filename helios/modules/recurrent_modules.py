import torch
import torch.nn as nn
from funcsigs import signature
from helios.modules.layers import BaseProcessingNet, FCBlock
from helios.modules.losses import L2Loss
from helios.modules.variational_inference import stack
from helios.utils.general_utils import AttrDict, batchwise_assign, map_dict, \
    concat_inputs, listdict2dictlist, subdict
from helios.utils.general_utils import broadcast_final


# Note: this post has an example custom implementation of LSTM from which we can derive a ConvLSTM/TreeLSTM
# https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/



class CustomLSTM(nn.Module):
    def __init__(self, cell):
        super(CustomLSTM, self).__init__()
        self.cell = cell
    
    def forward(self, inputs, length, initial_inputs=None, static_inputs=None):
        """
        
        :param inputs: These are sliced by time. Time is the second dimension
        :param length: Rollout length
        :param initial_inputs: These are not sliced and are overridden by cell output
        :param static_inputs: These are not sliced and can't be overridden by cell output
        :return:
        """
        # NOTE! Unrolling the cell directly will result in crash as the hidden state is not being reset
        # Use this function or CustomLSTMCell.unroll if needed
        initial_inputs, static_inputs = self.assert_begin(inputs, initial_inputs, static_inputs)

        step_inputs = initial_inputs.copy()
        step_inputs.update(static_inputs)
        lstm_outputs = []
        for t in range(length):
            step_inputs.update(map_dict(lambda x: x[:, t], inputs))  # Slicing
            output = self.cell(**step_inputs)
            
            self.assert_post(output, inputs, initial_inputs, static_inputs)
            # TODO Test what signature does with *args
            autoregressive_output = subdict(output, output.keys() & signature(self.cell.forward).parameters)
            step_inputs.update(autoregressive_output)
            lstm_outputs.append(output)
        
        # TODO recursively stack outputs
        lstm_outputs = listdict2dictlist(lstm_outputs)
        lstm_outputs = map_dict(lambda x: stack(x, dim=1), lstm_outputs)
            
        self.cell.reset()
        return lstm_outputs
    
    @staticmethod
    def assert_begin(inputs, initial_inputs, static_inputs):
        initial_inputs = initial_inputs or AttrDict()
        static_inputs = static_inputs or AttrDict()
        assert not (static_inputs.keys() & inputs.keys()), 'Static inputs and inputs overlap'
        assert not (static_inputs.keys() & initial_inputs.keys()), 'Static inputs and initial inputs overlap'
        assert not (inputs.keys() & initial_inputs.keys()), 'Inputs and initial inputs overlap'
        
        return initial_inputs, static_inputs
    
    @staticmethod
    def assert_post(output, inputs, initial_inputs, static_inputs):
        assert initial_inputs.keys() <= output.keys(), 'Initial inputs are not overridden'
        assert not ((static_inputs.keys() | inputs.keys()) & (output.keys())), 'Inputs are overridden'


class CustomGRU(nn.Module):
    def __init__(self, cell):
        super(CustomGRU, self).__init__()
        self.cell = cell
    
    def forward(self, inputs, length, initial_inputs=None, static_inputs=None):
        """
        :param inputs: Dictionary of input tensors sliced over time (time is the second dimension)
        :param length: Rollout length (sequence length)
        :param initial_inputs: Inputs that are not sliced and are overridden by cell output
        :param static_inputs: Inputs that are not sliced and cannot be overridden by cell output
        :return: Dictionary of outputs stacked over the time dimension
        """
        # NOTE: Unrolling the cell directly may result in a crash if the hidden state is not reset
        initial_inputs, static_inputs = self.assert_begin(inputs, initial_inputs, static_inputs)

        step_inputs = initial_inputs.copy()
        step_inputs.update(static_inputs)
        gru_outputs = []
        for t in range(length):
            # Slicing inputs at the current time step
            step_inputs.update(map_dict(lambda x: x[:, t], inputs))
            output = self.cell(**step_inputs)
            
            self.assert_post(output, inputs, initial_inputs, static_inputs)
            # Preparing inputs for the next time step
            autoregressive_output = subdict(
                output, output.keys() & signature(self.cell.forward).parameters
            )
            step_inputs.update(autoregressive_output)
            gru_outputs.append(output)
        
        # Stacking outputs over the time dimension
        gru_outputs = listdict2dictlist(gru_outputs)
        gru_outputs = map_dict(lambda x: stack(x, dim=1), gru_outputs)
            
        self.cell.reset()
        return gru_outputs
    
    @staticmethod
    def assert_begin(inputs, initial_inputs, static_inputs):
        initial_inputs = initial_inputs or AttrDict()
        static_inputs = static_inputs or AttrDict()
        assert not (static_inputs.keys() & inputs.keys()), 'Static inputs and inputs overlap'
        assert not (static_inputs.keys() & initial_inputs.keys()), 'Static inputs and initial inputs overlap'
        assert not (inputs.keys() & initial_inputs.keys()), 'Inputs and initial inputs overlap'
        
        return initial_inputs, static_inputs
    
    @staticmethod
    def assert_post(output, inputs, initial_inputs, static_inputs):
        assert initial_inputs.keys() <= output.keys(), 'Initial inputs are not overridden'
        assert not ((static_inputs.keys() | inputs.keys()) & output.keys()), 'Inputs are overridden'


class BaseProcessingLSTM(CustomLSTM):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(CustomLSTMCell(hp, in_dim, out_dim))
        
    def forward(self, input):
        """
        :param input: tensor of shape batch x time x channels
        :return:
        """
        return super().forward(AttrDict(cell_input=input), length=input.shape[1]).output


class BaseProcessingGRU(CustomGRU):
    '''
    Implementation of GRU cells
    '''
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(cell=CustomGRUCell(hp, in_dim, out_dim))
        
    def forward(self, input):
        """
        :param input: tensor of shape batch x time x channels
        :return:
        """
        return super().forward(AttrDict(cell_input=input), length=input.shape[1]).output

class BaseCell(nn.Module):
    @staticmethod
    def unroll_lstm(lstm, step_fn, time):
        # NOTE! The CustomLSTM class should be used instead of this direct interface in most cases
        lstm_outputs = [step_fn(t) for t in range(time)]
        lstm.reset()
        return lstm_outputs
    
    def make_lstm(self):
        return CustomLSTM(self)


class BaseCellGRU(nn.Module):
    @staticmethod
    def unroll_gru(gru, step_fn, time):
        # NOTE! The CustomGRU class should be used instead of this direct interface in most cases
        gru_outputs = [step_fn(t) for t in range(time)]
        gru.reset()
        return gru_outputs
    
    def make_gru(self):
        return CustomGRU(self)

# -- modified --

class CustomGRUCell(BaseCellGRU):
    def __init__(self, hp, input_size, output_size):
        """ A GRUCell wrapper """
        super(CustomGRUCell, self).__init__()
        
        hidden_size = hp.nz_mid_gru  # Adjusted parameter name # TODO: from SkillPriorMdl 128
        n_layers = hp.n_gru_layers   # Adjusted parameter name # TODO: from SkillPriorMdl 2
        
        self._hp = hp
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([
            nn.GRUCell(hidden_size, hidden_size) for _ in range(self.n_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.reset()
        self.init_bias(self.gru)
        
    @staticmethod
    def init_bias(gru):
        """ Initialize biases for GRU cells """
        for layer in gru:
            for name, param in layer.named_parameters():
                if "bias" in name:
                    n = param.size(0)
                    hidden_size = n // 3  # GRU has 3 gates: update, reset, new
                    # Bias order in PyTorch GRUCell: [bias_z | bias_r | bias_n]
                    # Initialize reset gate biases to 1
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
                    # Optionally, initialize update gate biases to -1
                    # param.data[0:hidden_size].fill_(-1.0)
    
    def reset(self):
        """ Reset the hidden states """
        batch_size = self._hp.batch_size
        device = self._hp.device
        self.hidden = [
            torch.zeros(batch_size, self.hidden_size, device=device) 
            for _ in range(self.n_layers)
        ]
        
    def get_state_size(self):
        """ Return the total size of the hidden states """
        return self.hidden_size * self.n_layers
        
    def var2state(self, var):
        """ Convert a tensor to a list representing the hidden states of the GRU """
        return list(torch.chunk(var, self.n_layers, dim=1))
        
    def state2var(self, state):
        """ Convert the list of hidden states to one tensor """
        return torch.cat(state, dim=1)
        
    def forward(self, *cell_input, **cell_kwinput):
        """
        At every time-step, the input to the GRU is a tuple of inputs
        :param cell_input: Positional inputs
        :param cell_kwinput: Keyword inputs
        :return: Output dictionary containing 'output'
        """
        if cell_kwinput:
            cell_input = cell_input + tuple(cell_kwinput.values())
        
        # Concatenate all inputs
        cell_input = concat_inputs(*cell_input)
        batch_size = cell_input.size(0)
        inp_extra_dim = list(cell_input.shape[2:])  # Trailing dimensions
        
        # Flatten input and apply embedding
        input_flat = cell_input.view(batch_size, -1)
        embedded = self.embed(input_flat)
        h_in = embedded
        
        # Pass through GRU layers
        for i in range(self.n_layers):
            hx = self.hidden[i]
            hx = self.gru[i](h_in, hx)
            self.hidden[i] = hx  # Update hidden state
            h_in = hx
        
        # Compute output
        output = self.output_layer(h_in)
        output = output.view(batch_size, *inp_extra_dim, self.output_size)
        return AttrDict(output=output)
        
    @property
    def hidden_var(self):
        """ Convert the list of hidden states to one tensor """
        return self.state2var(self.hidden)
        
    @hidden_var.setter
    def hidden_var(self, var):
        """ Convert the tensor to a list of hidden states """
        self.hidden = self.var2state(var)



class MaskedProcessingLSTM(CustomLSTM):
    """Sequence processing LSTM that maskes the hidden state based on mask input!"""
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(MaskedLSTMCell(hp, in_dim, out_dim))

    def forward(self, input, mask):
        return super().forward(AttrDict(cell_input=input, mask=mask), length=input.shape[1]).output


class BareProcessingLSTM(CustomLSTM):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__(BareLSTMCell(hp, in_dim, out_dim))

    def forward(self, input, hidden_state, length=None):
        """
        :param input: tensor of shape batch x time x channels
        :return:
        """
        if length is None: length = input.shape[1]
        initial_state = AttrDict(hidden_state=hidden_state)
        outputs = super().forward(AttrDict(cell_input=input), length=length, initial_inputs=initial_state)
        return outputs


class BidirectionalLSTM(nn.Module):
    def __init__(self, hp, in_dim, out_dim):
        super().__init__()
        self.forward_lstm = CustomLSTM(CustomLSTMCell(hp, in_dim, out_dim))
        self.backward_lstm = CustomLSTM(CustomLSTMCell(hp, out_dim, out_dim))

    def forward(self, input):
        input_length = input.shape[1]

        def apply_and_reverse(lstm, input):
            return lstm.forward(AttrDict(cell_input=input), length=input_length).output.flip([1])

        return apply_and_reverse(self.backward_lstm, apply_and_reverse(self.forward_lstm, input))



class CustomLSTMCell(BaseCell):
    def __init__(self, hp, input_size, output_size):
        """ An LSTMCell wrapper """
        super(CustomLSTMCell, self).__init__()
        
        hidden_size = hp.nz_mid_lstm
        n_layers = hp.n_lstm_layers
        
        # TODO make this a param dict
        self._hp = hp
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        # TODO use the LSTM class
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Linear(hidden_size, output_size)
        self.reset()
        self.init_bias(self.lstm)
        
    @staticmethod
    def init_bias(lstm):
        for layer in lstm:
            for param in filter(lambda p: "bias" in p[0], layer.named_parameters()):
                name, bias = param
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def reset(self):
        # TODO make this trainable
        self.hidden_var = torch.zeros(self._hp.batch_size, self.get_state_size(), device=self._hp.device)
        
    def get_state_size(self):
        return self.hidden_size * self.n_layers * 2
    
    def var2state(self, var):
        """ Converts a tensor to a list of tuples that represents the state of the LSTM """
        
        var_layers = torch.chunk(var, self.n_layers, 1)
        return [torch.chunk(layer, 2, 1) for layer in var_layers]
    
    def state2var(self, state):
        """ Converts the state of the LSTM to one tensor"""
        
        layer_tensors = [torch.cat(layer, 1) for layer in state]
        return torch.cat(layer_tensors, 1)
    
    def forward(self, *cell_input, **cell_kwinput):
        """
        at every time-step the input to the dense-reconstruciton LSTM is a tuple of (last_state, e_0, e_g)
        :param cell_input:
        :param reset_indicator:
        :return:
        """
        # TODO allow ConvLSTM
        if cell_kwinput:
            cell_input = cell_input + list(zip(*cell_kwinput.items()))[1]
        
        cell_input = concat_inputs(*cell_input)
        inp_extra_dim = list(cell_input.shape[2:])  # This keeps trailing dimensions (should be all shape 1)
        embedded = self.embed(cell_input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            # for LSTM --> hx, cx = lstm(in, (hx, cx))
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        output = self.output(h_in)
        return AttrDict(output=output.view(list(output.shape) + inp_extra_dim))
    
    @property
    def hidden_var(self):
        return self.state2var(self.hidden)
    
    @hidden_var.setter
    def hidden_var(self, var):
        self.hidden = self.var2state(var)


class BareLSTMCell(CustomLSTMCell):
    """Exposes hidden state, takes initial hidden state input, returns final hidden state."""

    def forward(self, *cell_input, **cell_kwinput):
        assert 'hidden_state' in cell_kwinput   # BareLSTMCell needs hidden state input
        self.hidden_var = cell_kwinput.pop('hidden_state')
        output = super().forward(*cell_input, **cell_kwinput)
        output.hidden_state = self.hidden_var
        return output


class ForwardLSTMCell(CustomLSTMCell):
    """Exposes hidden state, takes initial hidden state input, returns final hidden state."""

    def forward(self, x_t, *cell_input, **cell_kwinput):
        lsmt_output = super().forward(x_t, *cell_input, **cell_kwinput)
        return AttrDict(x_t=lsmt_output.output)


class MaskedLSTMCell(CustomLSTMCell):
    """Erases hidden cell state based on mask input."""

    def forward(self, *cell_input, **cell_kwinput):
        assert 'mask' in cell_kwinput   # MaskedLSTMCell needs mask as (keyword) input
        mask = cell_kwinput.pop('mask')
        self.hidden_var = self.hidden_var * mask[:, None]
        return super().forward(*cell_input, **cell_kwinput)


class RecurrentPredictor(nn.Module):
    """Recurrent forward prediction module."""
    def __init__(self, hp, input_size, output_size):
        super().__init__()
        self._hp = hp
        self.cell = ForwardLSTMCell(hp, input_size, output_size)
        self.lstm = CustomLSTM(self.cell)

    def forward(self, lstm_initial_inputs, steps, lstm_inputs=None, lstm_static_inputs=None, lstm_hidden_init=None):
        if lstm_inputs is None:
            lstm_inputs = {}
        if lstm_hidden_init is not None:
            self.cell.hidden_var = lstm_hidden_init     # initialize hidden state of LSTM if given
        lstm_outputs = self.lstm(lstm_inputs, steps, lstm_initial_inputs, lstm_static_inputs)
        return AttrDict(pred=lstm_outputs.x_t)


class RecBase(nn.Module):
    """ Base module for dense reconstruction. Handles skip connections loss, and action decoding
    
    """
    def __init__(self, hp, decoder):
        super().__init__()
        self._hp = hp
        self.decoder = decoder

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _dense_decode(self, inputs, encodings, seq_len):
        return self.decoder.decode_seq(inputs, encodings)

    def loss(self, inputs, model_output, extra_action=True, first_image=True):
        dense_losses = AttrDict()

        loss_gt = inputs.demo_seq
        loss_pad_mask = inputs.pad_mask
        actions_pad_mask = inputs.pad_mask[:, :-1]
        loss_actions = model_output.actions
        if not first_image:
            loss_gt = loss_gt[:, 1:]
            loss_pad_mask = loss_pad_mask[:, 1:]
        if extra_action:
            loss_actions = loss_actions[:, :-1]
            
        dense_losses.dense_img_rec = L2Loss(self._hp.dense_img_rec_weight, breakdown=1)\
            (model_output.images, loss_gt, weights=broadcast_final(loss_pad_mask, inputs.demo_seq))

        if self._hp.regress_actions:
            dense_losses.dense_action_rec = L2Loss(self._hp.dense_action_rec_weight)\
                (loss_actions, inputs.actions, weights=broadcast_final(actions_pad_mask, inputs.actions))

        return dense_losses


class LSTMCellInitializer(nn.Module):
    """Base class for initializing LSTM states for start and end node."""
    def __init__(self, hp, cell):
        super().__init__()
        self._hp = hp
        self._cell = cell
        self._hidden_size = self._cell.get_state_size()

    def forward(self, *inputs):
        raise NotImplementedError


class ZeroLSTMCellInitializer(LSTMCellInitializer):
    """Initializes hidden to constant 0."""
    def forward(self, *inputs):
        def get_init_hidden():
            return inputs[0].new_zeros((inputs[0].shape[0], self._hidden_sz))
        return get_init_hidden(), get_init_hidden()


class MLPLSTMCellInitializer(LSTMCellInitializer):
    """Initializes hidden with MLP that gets start and goal image encodings as input."""
    def __init__(self, hp, cell, input_sz):
        super().__init__(hp, cell)
        from helios.modules.subnetworks import Predictor    # to avoid cyclic import
        self.net = Predictor(self._hp, input_sz, output_size=2 * self._hidden_size, spatial=False,
                             num_layers=self._hp.init_mlp_layers, mid_size=self._hp.init_mlp_mid_sz)

    def forward(self, *inputs):
        hidden = self.net(*inputs)
        return hidden[:, :self._hidden_size], hidden[:, self._hidden_size:]





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import rlkit.torch.pytorch_util as ptu
# import numpy as np


# class GruAttentionEncoder(nn.Module):
#     def __init__(self, input_size, output_size, time_steps, encoding_mode):
#         super().__init__()
#         # Note: encoding mode is automatically set to trajectory
#         self.input_size = input_size // time_steps
#         self.output_size = output_size
#         self.time_steps = time_steps
#         self.encoding_mode = encoding_mode

#         self.gru = nn.GRU(self.input_size, self.output_size, batch_first=True)
#         self.attn = nn.Linear(self.output_size, self.output_size)

#     def forward(self, x):
#         hidden = self.initHidden(x.size(0))

#         x = x.view(x.size(0), self.time_steps, self.input_size)
#         _, hidden = self.gru(x, hidden)

#         hidden = hidden.squeeze(0)

#         # TODO: Test
#         # attn_weights = F.softmax(self.attn(hidden), dim=-1)
#         # hidden = attn_weights * hidden

#         hidden = F.relu(hidden)
#         return hidden

#     def initHidden(self, b_size):
#         return ptu.zeros(1, b_size, self.output_size)