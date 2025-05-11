import torch
import torch.nn as nn
import numpy as np

class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # mLSTM components
        self.mlstm_conv = nn.Conv1d(input_size, input_size, kernel_size=4, padding=1)
        self.mlstm_qkv = nn.Linear(input_size, 3 * input_size)
        self.mlstm_proj = nn.Linear(input_size, hidden_size)
        
        # sLSTM components
        self.slstm_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=4, padding=1)
        self.slstm_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.slstm_proj = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanisms
        self.mlstm_gate = nn.Linear(input_size + hidden_size, input_size)
        self.slstm_gate = nn.Linear(hidden_size + input_size, hidden_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x, h_m, h_s):
        # mLSTM path
        x_conv = self.mlstm_conv(x.transpose(1, 2)).transpose(1, 2)
        qkv = self.mlstm_qkv(x_conv)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Matrix memory update
        h_m = h_m + torch.matmul(q, k.transpose(-2, -1)) * v
        
        # sLSTM path
        h_conv = self.slstm_conv(h_s.transpose(1, 2)).transpose(1, 2)
        qkv_s = self.slstm_qkv(h_conv)
        q_s, k_s, v_s = qkv_s.chunk(3, dim=-1)
        
        # State update
        h_s = h_s + torch.matmul(q_s, k_s.transpose(-2, -1)) * v_s
        
        # Gating
        gate_m = torch.sigmoid(self.mlstm_gate(torch.cat([x, h_s], dim=-1)))
        gate_s = torch.sigmoid(self.slstm_gate(torch.cat([h_m, x], dim=-1)))
        
        # Final updates with gating
        h_m = gate_m * h_m + (1 - gate_m) * x
        h_s = gate_s * h_s + (1 - gate_s) * h_m
        
        # Layer normalization
        h_m = self.norm1(h_m)
        h_s = self.norm2(h_s)
        
        return h_m, h_s

def test_xlstm():
    # Test parameters
    batch_size = 4
    seq_len = 8
    input_size = 64
    hidden_size = 128
    
    # Create model
    xlstm = xLSTMBlock(input_size, hidden_size)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, input_size)
    h_m = torch.zeros(batch_size, seq_len, input_size)
    h_s = torch.zeros(batch_size, seq_len, hidden_size)
    
    # Forward pass
    h_m_out, h_s_out = xlstm(x, h_m, h_s)
    
    # Verify shapes
    assert h_m_out.shape == (batch_size, seq_len, input_size)
    assert h_s_out.shape == (batch_size, seq_len, hidden_size)
    
    # Test memory retention
    x2 = torch.randn(batch_size, seq_len, input_size)
    h_m_out2, h_s_out2 = xlstm(x2, h_m_out, h_s_out)
    
    # Verify that states are different after new input
    assert not torch.allclose(h_m_out, h_m_out2)
    assert not torch.allclose(h_s_out, h_s_out2)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_xlstm() 