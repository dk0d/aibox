# AIBox

My AI toolbox of helper functions and other assorted utils

# Configuration Folder Info

- Model configurations go into `models` folder
- Experiment configurations go into `experiments` folder
- Configuration folders can have subdirectories
  - for example: a model at the path `models/cnn/classifier1.toml` has the model configuration name `cnn/classifier1`
- any `default.toml` in a folder or subfolder will be loaded first _before_ the specified configuration name
  - the specified config will override any parameters specified with the same key path from the default config

> Refer to TOML spec for help on syntax: [TOML.io](https://toml.io/en/)

# Project Structure

```bash
├── configs
│ ├── experiments
│ │  ├── vae.toml
│ │  ├── default.toml
│ │  ├── mnist.toml
│ ├── models
│ │  ├── ae/vae.toml
│ │  ├── pix2pix.toml
│ │  └── default.toml
├── src
│  └── project_package
├── pyproject.toml
└── README.md
```

- configs folder can be anywhere, can override via CLI args
- can specify multiple configurations and `OmegaConf` merges them

# Examples

Models

```toml
[model]
_target_ = "project.models.CNN" # required
in_channels = 3

[[model.optimizers]] # makes an array of objects at the models.args.optimizers path
_target_ = 'torch.optim.Adam'
lr = 1e-3


```

Experiment

```toml

[data]
_target_ = "torch"

[data.args]
# ...

[loss]

[trainer]

[tuner] # optional params for ray tune

[tuner.values]
param.values = [0, 1, 28]
param.search_space = 'choice' # see raytune for search_space names

```
