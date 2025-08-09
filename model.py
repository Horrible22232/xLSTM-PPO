import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig
from typing import Optional, Tuple, Dict


class xLSTM(nn.Module):
    """xLSTM recurrent module adapted for RL usage.

    Uses a single sLSTM block with conv1d disabled (kernel_size=0) so the
    recurrent state consists solely of the sLSTM state tensors.

    State format (no conv state):
    - Batched state (step mode): {"block_0": {"slstm_state": Tensor(4, B, H)}}
    - Per-sequence state list (sequence mode): list of length S where each item is
      {"block_0": {"slstm_state": Tensor(4, 1, H)}}
    """

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4, dropout: float = 0.0, backend: str = "vanilla"):
        """Initialize the xLSTM wrapper used in RL.

        Arguments:
            input_size {int} -- Feature dimension of the input per time step
            hidden_size {int} -- Size of the sLSTM hidden embedding H
            num_heads {int} -- Number of attention heads inside the sLSTM layer
            dropout {float} -- Dropout probability used inside the sLSTM block
            backend {str} -- Implementation backend: "vanilla" (PyTorch) or "cuda" (custom CUDA kernels)

        Notes:
            - Convolutional state is disabled (conv1d_kernel_size=0), so the recurrent state
              only consists of the sLSTM state tensors (shape (4, B, H)).
            - Step mode expects a batched state dict; sequence mode in training accepts a list
              of per-sequence state dicts and handles merging/splitting internally.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backend = backend

        self.input_proj = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size)

        slstm_layer_cfg = sLSTMLayerConfig(
            embedding_dim=hidden_size,
            num_heads=num_heads,
            conv1d_kernel_size=0,  # disable conv state to simplify external state handling
            dropout=dropout,
            backend=backend,
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
        """Creates a zero-initialized xLSTM state for a batched step.

        Arguments:
            batch_size {int} -- Batch size B for the state
            device {torch.device} -- Target device
            dtype {torch.dtype} -- Data type of the created tensors. Defaults to torch.float32

        Returns:
            {dict} -- {"block_0": {"slstm_state": Tensor(4, B, H)}} where 4 = number of sLSTM states
        """
        # sLSTMCellConfig default num_states is 4
        slstm_state = torch.zeros((4, batch_size, self.hidden_size), dtype=dtype, device=device)
        # conv_state is not used when conv1d_kernel_size=0
        return {"block_0": {"slstm_state": slstm_state}}

    def step(self, x: torch.Tensor, state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        """Single-step forward.

        Arguments:
            x {torch.Tensor} -- Input of shape (B, 1, input_size)
            state {dict | None} -- Batched state dict {"block_0": {"slstm_state": Tensor(4, B, H)}}

        Returns:
            {torch.Tensor} -- Output y of shape (B, 1, H)
            {dict} -- Updated batched state dict in the same format as input
        """
        x_proj = self.input_proj(x)
        y, state = self.stack.step(x_proj, state=state)
        return y, state

    def forward_sequence(self, x: torch.Tensor, state: Optional[Dict[str, Dict[str, torch.Tensor]] | list]) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]] | list]:
        """Sequence forward over S time steps.

        Supports two state formats:
            - Batched dict (step mode): {"block_0": {"slstm_state": Tensor(4, B, H)}}
            - List (per-sequence mode, used with padding): list[dict], each with
              {"block_0": {"slstm_state": Tensor(4, 1, H)}}

        Arguments:
            x {torch.Tensor} -- Input tensor (B, S, input_size)
            state {dict | list | None} -- Batched dict or list of per-sequence dicts as above

        Returns:
            {torch.Tensor} -- Output y of shape (B, S, H)
            {dict | list} -- If input was list, returns list of per-sequence dicts; else returns batched dict
        """
        B, S, _ = x.shape

        # If state is a list of per-sequence dicts, merge into a single batched dict
        def _merge_state(state_list: list[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
            merged: Dict[str, Dict[str, torch.Tensor]] = {}
            for block_key in state_list[0].keys():
                merged[block_key] = {}
                for state_key in state_list[0][block_key].keys():
                    tensors = [sd[block_key][state_key] for sd in state_list if sd[block_key][state_key] is not None]
                    if len(tensors) == 0:
                        merged[block_key][state_key] = None
                    else:
                        merged[block_key][state_key] = torch.cat(tensors, dim=1)  # (num_states, B, H)
            return merged

        def _split_state(merged_state: Dict[str, Dict[str, torch.Tensor]]) -> list[Dict[str, Dict[str, torch.Tensor]]]:
            out: list[Dict[str, Dict[str, torch.Tensor]]] = []
            for b in range(B):
                d: Dict[str, Dict[str, torch.Tensor]] = {}
                for block_key, block_state in merged_state.items():
                    d[block_key] = {}
                    for state_key, tensor in block_state.items():
                        if tensor is None:
                            d[block_key][state_key] = None
                        else:
                            d[block_key][state_key] = tensor[:, b : b + 1, :].contiguous()
                out.append(d)
            return out

        batched_state = _merge_state(state) if isinstance(state, list) else state

        outputs = []
        cur_state = batched_state
        for t in range(S):
            y_t, cur_state = self.step(x[:, t : t + 1, :], cur_state)
            outputs.append(y_t)
        y = torch.cat(outputs, dim=1)

        return y, (_split_state(cur_state) if isinstance(state, list) else cur_state)

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
            xlstm_backend = self.recurrence.get("xlstm_backend", "cuda" if torch.cuda.is_available() else "vanilla")
            self.recurrent_layer = xLSTM(
                input_size=in_features_next_layer,
                hidden_size=self.recurrence["hidden_state_size"],
                num_heads=4,
                dropout=0.0,
                backend=xlstm_backend,
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

    def forward(self, obs: torch.Tensor, recurrent_cell: torch.tensor, device: torch.device, sequence_length: int = 1):
        """Forward pass of the model

        Arguments:
            obs {torch.Tensor} -- Batch of observations
            recurrent_cell -- Recurrent state structure depends on layer type:
                - GRU: Tensor (1, B, H)
                - LSTM: Tuple[Tensor, Tensor] both (1, B, H)
                - xLSTM:
                    - sequence_length == 1: Dict {"block_0": {"slstm_state": Tensor(4, B, H)}}
                    - sequence_length > 1: list of length (num_sequences) with items
                      Dict {"block_0": {"slstm_state": Tensor(4, 1, H)}}
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {list[Categorical]} -- Policy branches (multi-discrete)
            {torch.Tensor} -- Value Function: shape (B,)
            Recurrent cell -- Same structure as input for step mode; for xLSTM in
                sequence mode the input recurrent_cell is returned unchanged.
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
                # Batched forward over all sequences; accepts list of per-sequence states
                h, _ = self.recurrent_layer.forward_sequence(h, state=recurrent_cell)
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