# AIBox

My AI toolbox of helper functions and other assorted utils

# Configuration Folder Info

- Model configurations go into `models` folder
- Experiment configurations go into `experiments` folder
- Configuration folders can have subdirectories
  - for example: a model at the path `models/cnn/classifier1.toml` has the model configuration name `cnn/classifier1`
- any `default.toml` in a folder or subfolder will be loaded first _before_ the specified configuration name
  - the specified config will override any parameters specified with the same key path from the default config
- the `model`, `data`,

> Refer to TOML spec for help on syntax: [TOML.io](https://toml.io/en/)

# Examples

Models

```toml
[model]
class_path = "project.models.CNN" # required

[model.args]
in_channels = 3

[[model.args.optimizers]] # makes an array of objects at the models.args.optimizers path
class_path = 'torch.optim.Adam'
args.lr = 1e-3


```

or

```toml
[model]
class_path = "project.models.CNN" # required
args.in_channels = 3

[[model.args.optimizers]] # makes an array of objects at the models.args.optimizers path
class_path = 'torch.optim.Adam'
args.lr = 1e-3

```

Experiment

```toml

[data]
class_path = "torch"

[data.args]
# ...

[loss]

[trainer]

[tuner] # optional params for ray tune


[tuner.values]

param.values = [0, 1, 28]
param.search_space = 'choice' # see raytune for search_space names

```
