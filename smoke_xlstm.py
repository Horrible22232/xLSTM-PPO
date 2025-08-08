import torch
from yaml_parser import YamlParser
from utils import create_env
from model import ActorCriticModel


def main():
    cfg = YamlParser('./configs/cartpole_xlstm.yaml').get_config()

    env = create_env(cfg['environment'])
    obs_space = env.observation_space
    action_space = env.action_space
    if hasattr(action_space, 'n'):
        action_shape = (action_space.n,)
    else:
        action_shape = tuple(action_space.nvec)

    model = ActorCriticModel(cfg, obs_space, action_space_shape=action_shape)
    model.eval()

    device = torch.device('cpu')
    if cfg['recurrence']['layer_type'] == 'gru':
        hxs, _ = model.init_recurrent_cell_states(2, device)
        rc = hxs
    elif cfg['recurrence']['layer_type'] == 'lstm':
        hxs, cxs = model.init_recurrent_cell_states(2, device)
        rc = (hxs, cxs)
    else:  # xlstm
        state_dict, _ = model.init_recurrent_cell_states(2, device)
        rc = state_dict

    # single-step forward
    obs = torch.zeros((2,) + obs_space.shape, dtype=torch.float32)
    pi, val, rc_out = model(obs, rc, device)
    assert isinstance(pi, list) and len(pi) == len(action_shape)
    assert val.shape == (2,)

    # sequence forward using configured sequence_length
    seq_len = cfg['recurrence']['sequence_length']
    obs2 = torch.zeros((2 * seq_len,) + obs_space.shape, dtype=torch.float32)
    if cfg['recurrence']['layer_type'] == 'xlstm':
        # turn batch state dict into list of per-sequence state dicts
        rc_list = []
        for i in range(2):
            slstm_state = rc_out['block_0']['slstm_state'][:, i:i+1, :].contiguous()
            rc_list.append({'block_0': {'slstm_state': slstm_state}})
        rc2 = rc_list
    else:
        rc2 = rc
    pi2, val2, _ = model(obs2, rc2, device, seq_len)
    assert isinstance(pi2, list) and len(pi2) == len(action_shape)
    assert val2.shape == (2 * seq_len,)

    env.close()
    print('SMOKE_OK')


if __name__ == '__main__':
    main()



