import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from typing import Optional, Tuple, Dict


class RLxLSTM(nn.Module):
    """Wrapper around the repository's xLSTM to support RL recurrent usage.

    We use a single sLSTM block with conv1d disabled (kernel_size=0) so the
    recurrent state consists solely of the sLSTM state tensors.
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_proj = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size)

        slstm_layer_cfg = sLSTMLayerConfig(
            embedding_dim=hidden_size,
            num_heads=num_heads,
            conv1d_kernel_size=0,  # disable conv state to simplify external state handling
            dropout=dropout,
            backend="vanilla",
        )
        slstm_block_cfg = sLSTMBlockConfig(slstm=slstm_layer_cfg, feedforward=None)

        stack_cfg = xLSTMBlockStackConfig(
            mlstm_block=None,
            slstm_block=slstm_block_cfg,
            context_length=-1,
            num_blocks=1,
            embedding_dim=hidden_size,
            add_post_blocks_norm=True,
            bias=False,
            dropout=dropout,
            slstm_at="all",
        )
        self.stack = xLSTMBlockStack(config=stack_cfg)

    @torch.no_grad()
    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Dict[str, Dict[str, torch.Tensor]]:
        # sLSTMCellConfig default num_states is 4
        slstm_state = torch.zeros((4, batch_size, self.hidden_size), dtype=dtype, device=device)
        # conv_state is not used when conv1d_kernel_size=0
        return {"block_0": {"slstm_state": slstm_state}}

    def step(self, x: torch.Tensor, state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        # x: (B, 1, input_size)
        x_proj = self.input_proj(x)
        y, state = self.stack.step(x_proj, state=state)
        return y, state

    def forward_sequence(self, x: torch.Tensor, state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        # x: (B, S, input_size); step through to respect provided initial state
        B, S, _ = x.shape
        outputs = []
        cur_state = state
        for t in range(S):
            y_t, cur_state = self.step(x[:, t : t + 1, :], cur_state)
            outputs.append(y_t)
        y = torch.cat(outputs, dim=1)
        return y, cur_state

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        # Recurrent layer (GRU, LSTM, or xLSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "xlstm":
            self.recurrent_layer = RLxLSTM(
                input_size=in_features_next_layer,
                hidden_size=self.recurrence["hidden_state_size"],
                num_heads=4,
                dropout=0.0,
            )
        # Init recurrent layer
        if self.recurrence["layer_type"] != "xlstm":
            for name, param in self.recurrent_layer.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, np.sqrt(2))
        
        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, recurrent_cell:torch.tensor, device:torch.device, sequence_length:int=1):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Forward recurrent layer (GRU, LSTM, or xLSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            if self.recurrence["layer_type"] == "xlstm":
                x_step = h.unsqueeze(1)
                y, recurrent_cell = self.recurrent_layer.step(x_step, state=recurrent_cell)
                h = y.squeeze(1)
            else:
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
                h = h.squeeze(1)  # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent layer
            if self.recurrence["layer_type"] == "xlstm":
                # recurrent_cell is expected to be a list of per-sequence state dicts
                num_sequences = h.shape[0]
                outputs = []
                for seq_idx in range(num_sequences):
                    y_seq, _ = self.recurrent_layer.forward_sequence(
                        h[seq_idx : seq_idx + 1], state=recurrent_cell[seq_idx]
                    )
                    outputs.append(y_seq)
                h = torch.cat(outputs, dim=0)
            else: # GRU or LSTM
                # These layers handle batch_first=True and initial hidden states correctly.
                h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # The output of the recurrent layer is not activated as it already utilizes its own activations.

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, recurrent_cell

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
 
    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        if self.recurrence["layer_type"] == "gru":
            hxs = torch.zeros((1, num_sequences, self.recurrence["hidden_state_size"]), dtype=torch.float32, device=device)
            return hxs, None
        elif self.recurrence["layer_type"] == "lstm":
            hxs = torch.zeros((1, num_sequences, self.recurrence["hidden_state_size"]), dtype=torch.float32, device=device)
            cxs = torch.zeros((1, num_sequences, self.recurrence["hidden_state_size"]), dtype=torch.float32, device=device)
            return hxs, cxs
        elif self.recurrence["layer_type"] == "xlstm":
            # Return xLSTM state dict for the whole batch
            return self.recurrent_layer.init_state(batch_size=num_sequences, device=device), None