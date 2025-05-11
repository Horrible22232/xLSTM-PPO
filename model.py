import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # mLSTM components
        self.mlstm_qkv = nn.Linear(input_size, 3 * input_size)
        
        # sLSTM components
        self.slstm_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.slstm_proj = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanisms
        self.mlstm_gate = nn.Linear(2 * hidden_size, hidden_size) # Input is cat(x_proj, h_s), both hidden_size 
        self.slstm_gate = nn.Linear(2 * hidden_size, hidden_size) # Input is cat(h_s_candidate, h_m), both hidden_size
        
        # Layer normalization
        self.norm_h_m = nn.LayerNorm(hidden_size)
        self.norm_h_s = nn.LayerNorm(hidden_size)
        
    def forward(self, x, h_m, h_s):
        # x shape: (batch, seq, input_size)
        # h_m shape: (batch, seq, hidden_size) - assuming after transpose
        # h_s shape: (batch, seq, hidden_size) - assuming after transpose

        # mLSTM path
        qkv_m = self.mlstm_qkv(x) 
        q_m, k_m, v_m = qkv_m.chunk(3, dim=-1)
        # attn_weights_m = torch.sigmoid(q_m * k_m) # Original test logic had this

        # sLSTM path
        qkv_s = self.slstm_qkv(h_s) 
        q_s, k_s, v_s = qkv_s.chunk(3, dim=-1)
        attn_weights_s = torch.sigmoid(q_s * k_s) 
        h_s_candidate = attn_weights_s * v_s 

        # Revised gating and updates for dimensional correctness
        if self.input_size != self.hidden_size:
            if not hasattr(self, 'input_to_hidden_proj') or self.input_to_hidden_proj.weight.device != x.device:
                # print(f"DEBUG: Creating/Recreating input_to_hidden_proj on device: {x.device}")
                self.input_to_hidden_proj = nn.Linear(self.input_size, self.hidden_size).to(x.device)
            x_proj = self.input_to_hidden_proj(x)
        else:
            x_proj = x

        # h_s update (influenced by its own path and h_m)
        # The slstm_gate takes h_s_candidate and x (or h_m)
        # Let's use h_m as per a common LSTM/GRU cross-influence pattern
        gate_s_input = torch.cat([h_s_candidate, h_m], dim=-1) 
        gate_s = torch.sigmoid(self.slstm_gate(gate_s_input)) 
        h_s = gate_s * h_s_candidate + (1 - gate_s) * h_m # h_m influences h_s. h_s gets updated first.

        # h_m update (influenced by x_proj and the new h_s)
        gate_m_input = torch.cat([x_proj, h_s], dim=-1) # h_s is new, x_proj is current input projected
        gate_m = torch.sigmoid(self.mlstm_gate(gate_m_input)) 
        
        h_m = gate_m * x_proj + (1 - gate_m) * h_m # Reverted to direct assignment
        
        h_m = self.norm_h_m(h_m)
        h_s = self.norm_h_s(h_s)
        
        return h_m, h_s

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
            self.recurrent_layer = xLSTMBlock(in_features_next_layer, self.recurrence["hidden_state_size"])
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
                h_m_in_orig, h_s_in_orig = recurrent_cell # (1, batch_size, hidden_size)
                
                x_for_block = h.unsqueeze(1) # (batch_size, 1, input_size)
                
                # Transpose h_m, h_s to (batch_size, 1, hidden_size) for xLSTMBlock
                h_m_for_block = h_m_in_orig.transpose(0, 1)
                h_s_for_block = h_s_in_orig.transpose(0, 1)
                
                h_m_out_block, h_s_out_block = self.recurrent_layer(x_for_block, h_m_for_block, h_s_for_block)
                # h_m_out_block, h_s_out_block are (batch_size, 1, hidden_size)
                
                h = h_s_out_block.squeeze(1)  # Use sLSTM output as main state for subsequent layers
                
                # Store back into recurrent_cell in original format (1, batch_size, hidden_size)
                recurrent_cell = (h_m_out_block.transpose(0, 1), h_s_out_block.transpose(0, 1))
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
                h_m_initial, h_s_initial = recurrent_cell
                
                # Transpose to (num_actor_sequences, 1, hidden_size)
                h_m_t = h_m_initial.transpose(0,1) 
                h_s_t = h_s_initial.transpose(0,1)

                outputs_s = []
                # Manual unrolling over the sequence
                for t in range(sequence_length):
                    x_t = h[:, t, :].unsqueeze(1) # (num_actor_sequences, 1, num_features)
                    h_m_t, h_s_t = self.recurrent_layer(x_t, h_m_t, h_s_t) # h_m_t, h_s_t are (num_actor_sequences, 1, hidden_size)
                    outputs_s.append(h_s_t) # Store the s_state for each step
                
                h = torch.cat(outputs_s, dim=1) # (num_actor_sequences, sequence_length, hidden_size)
                
                # Final recurrent_cell should be the last step's states, transposed back
                recurrent_cell = (h_m_t.transpose(0,1), h_s_t.transpose(0,1))
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
        elif self.recurrence["layer_type"] == "lstm" or self.recurrence["layer_type"] == "xlstm":
            # For xLSTM, hxs will be m_state, cxs will be s_state. Both are hidden_size.
            hxs = torch.zeros((1, num_sequences, self.recurrence["hidden_state_size"]), dtype=torch.float32, device=device)
            cxs = torch.zeros((1, num_sequences, self.recurrence["hidden_state_size"]), dtype=torch.float32, device=device)
            return hxs, cxs