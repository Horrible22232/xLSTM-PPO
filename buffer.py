from gymnasium import spaces
import torch
# import numpy as np # No longer needed if all are tensors

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, device:torch.device) -> None:
        """
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"]
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        hidden_state_size = config["recurrence"]["hidden_state_size"]
        self.layer_type = config["recurrence"]["layer_type"]
        self.sequence_length = config["recurrence"]["sequence_length"]
        self.true_sequence_length = 0

        # Initialize the buffer's data storage with PyTorch tensors
        self.rewards = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.bool)
        self.obs = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape, dtype=torch.float32) # Assuming obs are float32
        self.hxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=torch.float32)
        self.cxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=torch.float32)
        if self.layer_type == "xlstm":
            self.mxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=torch.float32)
            self.sxs = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size), dtype=torch.float32)
        self.log_probs = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.float32)
        self.values = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.float32)
        self.advantages = torch.zeros((self.n_workers, self.worker_steps), dtype=torch.float32)
        self.loss_mask_buffer = torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool) # Buffer for actual loss mask from dones

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        samples = {
            "obs": self.obs,
            "actions": self.actions,
            "loss_mask": torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool) # Initial full loss mask
        }
        
        if self.layer_type == "xlstm":
            samples["mxs"] = self.mxs
            samples["sxs"] = self.sxs
        else:
            samples["hxs"] = self.hxs
            if self.layer_type == "lstm":
                samples["cxs"] = self.cxs

        episode_done_indices = []
        for w in range(self.n_workers):
            worker_dones = self.dones[w].cpu() # Ensure on CPU for nonzero
            done_indices_for_worker = worker_dones.nonzero(as_tuple=False).squeeze().tolist()
            if not isinstance(done_indices_for_worker, list): # Handle single done case
                done_indices_for_worker = [done_indices_for_worker]
            if not done_indices_for_worker or done_indices_for_worker[-1] != self.worker_steps - 1:
                done_indices_for_worker.append(self.worker_steps - 1)
            episode_done_indices.append(done_indices_for_worker)

        sequences = []
        self.actual_sequence_length = self.sequence_length # Assuming fixed sequence length for now
        self.loss_mask_buffer.fill_(False) # Reset for correct mask generation

        for w in range(self.n_workers):
            start_idx = 0
            for done_idx in episode_done_indices[w]:
                trajectory_len = done_idx - start_idx + 1
                for seq_start_in_traj in range(0, trajectory_len, self.sequence_length):
                    buffer_seq_start = start_idx + seq_start_in_traj
                    buffer_seq_end = min(buffer_seq_start + self.sequence_length, start_idx + trajectory_len)
                    current_seq_len = buffer_seq_end - buffer_seq_start

                    seq_data = {}
                    seq_data["obs"] = self.obs[w, buffer_seq_start:buffer_seq_end]
                    seq_data["actions"] = self.actions[w, buffer_seq_start:buffer_seq_end]
                    seq_data["rewards"] = self.rewards[w, buffer_seq_start:buffer_seq_end]
                    seq_data["dones"] = self.dones[w, buffer_seq_start:buffer_seq_end]
                    seq_data["values"] = self.values[w, buffer_seq_start:buffer_seq_end]
                    seq_data["log_probs"] = self.log_probs[w, buffer_seq_start:buffer_seq_end]
                    seq_data["advantages"] = self.advantages[w, buffer_seq_start:buffer_seq_end]
                    
                    # Initial recurrent states for this sequence (from the start of the sequence)
                    if self.layer_type == "xlstm":
                        seq_data["mxs"] = self.mxs[w, buffer_seq_start:buffer_seq_start+1] # Take first step state
                        seq_data["sxs"] = self.sxs[w, buffer_seq_start:buffer_seq_start+1]
                    else:
                        seq_data["hxs"] = self.hxs[w, buffer_seq_start:buffer_seq_start+1]
                        if self.layer_type == "lstm":
                            seq_data["cxs"] = self.cxs[w, buffer_seq_start:buffer_seq_start+1]

                    # Pad if current_seq_len is less than self.sequence_length
                    padding_len = self.sequence_length - current_seq_len
                    current_loss_mask = torch.ones(current_seq_len, dtype=torch.bool)

                    if padding_len > 0:
                        for key in ["obs", "actions", "rewards", "dones", "values", "log_probs", "advantages"]:
                            padding_shape = (padding_len,) + seq_data[key].shape[1:]
                            padding_tensor = torch.zeros(padding_shape, dtype=seq_data[key].dtype)
                            seq_data[key] = torch.cat([seq_data[key], padding_tensor], dim=0)
                        padding_mask = torch.zeros(padding_len, dtype=torch.bool)
                        current_loss_mask = torch.cat([current_loss_mask, padding_mask], dim=0)
                    
                    seq_data["loss_mask"] = current_loss_mask
                    sequences.append(seq_data)
                start_idx = done_idx + 1
        
        self.num_sequences = len(sequences)

        # --- MORE AGGRESSIVE DEBUGGING for sequences list --- Block removed
        # --- END MORE AGGRESSIVE DEBUGGING ---

        if self.num_sequences == 0: 
            print("Warning: No sequences generated in buffer.prepare_batch_dict. This might indicate an issue.")
            # If sequences is empty, self.samples_flat will remain empty, which might cause issues later.
            # Consider how to handle this: maybe skip epoch, or ensure dummy data that doesn't break downstream.

        self.samples_flat = {}
        if sequences: 
            all_keys = sequences[0].keys() if sequences[0] else [] 
            if not all_keys:
                print("\n!!! DEBUG: sequences[0] has no keys (even though sequences list is not empty)!")

            for key in all_keys:
                list_to_cat = [] 
                try:
                    list_to_cat = [s[key] for s in sequences]
                    if key in ["hxs", "cxs", "mxs", "sxs"]: 
                        self.samples_flat[key] = torch.cat(list_to_cat, dim=0).squeeze(1)
                    else: 
                        self.samples_flat[key] = torch.cat(list_to_cat, dim=0)
                except TypeError as te:
                    print(f"\n!!! TypeError during torch.cat for key='{key}': {te}")
                    for i, item_in_list in enumerate(list_to_cat): 
                        print(f"    Item {i} in list for key '{key}': type={type(item_in_list)}")
                        if hasattr(item_in_list, 'dtype'):
                            print(f"        dtype: {item_in_list.dtype}")
                    raise te 
                except Exception as e:
                    print(f"\n!!! UNEXPECTED ERROR during torch.cat for key='{key}': {e}")
                    for i, item_in_list in enumerate(list_to_cat):
                        print(f"    Item {i} in list for key '{key}': type={type(item_in_list)}")
                    raise e
        else:
            print("\n!!! DEBUG: sequences list is empty before attempting to flatten! (This was already printed above)")

        # Calculate the average true sequence length for reporting
        if "loss_mask" in self.samples_flat and self.samples_flat["loss_mask"].numel() > 0 and self.num_sequences > 0:
            num_total_elements = self.samples_flat["loss_mask"].shape[0]
            # self.actual_sequence_length is the padded sequence length
            elements_per_padded_sequence = num_total_elements // self.num_sequences
            if elements_per_padded_sequence == self.actual_sequence_length:
                try:
                    reshaped_mask = self.samples_flat["loss_mask"].reshape(self.num_sequences, self.actual_sequence_length)
                    self.true_sequence_length = torch.mean(torch.sum(reshaped_mask, dim=1).float()).item()
                except Exception as e:
                    print(f"Error calculating true_sequence_length: {e}. Defaulting to padded length.")
                    self.true_sequence_length = float(self.actual_sequence_length)
            else:
                print(f"Shape mismatch for true_sequence_length. Total elements: {num_total_elements}, Num sequences: {self.num_sequences}, Padded length: {self.actual_sequence_length}. Defaulting to padded length.")
                self.true_sequence_length = float(self.actual_sequence_length)
        else:
            self.true_sequence_length = float(self.actual_sequence_length)
        # print(f"Buffer: True sequence length (avg): {self.true_sequence_length}, Padded sequence length: {self.actual_sequence_length}") -- Removed

    def recurrent_mini_batch_generator(self) -> dict:
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.
        
        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += 1 # Add the remainder if the sequence count and the number of mini batches do not share a common divider
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key in ["hxs", "cxs", "mxs", "sxs"]:
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
                else:
                    # Select padded data for all other keys (e.g., obs, actions, log_probs, advantages, values, loss_mask)
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.device)
            start = end
            yield mini_batch

    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = self.dones.logical_not()
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = self.rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]