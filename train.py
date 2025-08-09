import torch
import os
import shutil
from docopt import docopt
from trainer import PPOTrainer
from yaml_parser import YamlParser

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/cartpole_xlstm.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Configure xLSTM backend based on device and toolchain availability
    if config.get('recurrence', {}).get('layer_type') == 'xlstm':
        # Allow user override if already provided in config
        if 'xlstm_backend' not in config['recurrence']:
            # Env override to force CUDA backend attempt
            if os.environ.get('XLSTM_FORCE_CUDA_BACKEND', '0') == '1':
                use_cuda_backend = True
            else:
                use_cuda_backend = device.type == 'cuda'
            if use_cuda_backend:
                # On Windows, require MSVC cl.exe to build CUDA extensions; otherwise fall back
                if os.name == 'nt' and shutil.which('cl.exe') is None:
                    use_cuda_backend = False
                # If CUDA_HOME is not set on non-Windows, likely cannot build extensions
                if os.name != 'nt' and 'CUDA_HOME' not in os.environ:
                    use_cuda_backend = False
            config['recurrence']['xlstm_backend'] = 'cuda' if use_cuda_backend else 'vanilla'

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()