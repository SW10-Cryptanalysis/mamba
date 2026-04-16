# Mamba

#### Run ```uv sync --all-extras``` to install necessary packages. 
#### Must be run in Linux environment
#### To train model: 
``` shell
uv run -m src.train --resume [path_to_checkpoint]
```
--resume is an optional argument. If provided, training will resume from checkpoint provided (e.g. outputs/exp_{timestamp}/latest.pth). If --resume is provided but no path to checkpoint is provided, it will resume training from latest epoch from latest model.

#### To test model: 
``` shell
uv run -m src.eval [model]
```
[model] is an optional argument where you can specify a model. If not provided, the newest model will be used. 
#### To run tests model: 
``` shell
uv run pytest
```