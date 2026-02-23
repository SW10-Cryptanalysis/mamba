# Mamba

#### Run ```uv sync --all-extras``` to install necessary packages. 
#### Must be run in Linux environment
#### To train model: 
``` shell
uv run -m src.train
```
#### To test model: 
``` shell
uv run -m src.eval [model]
```
[model] is an optional argument where you can specify a model. If not provided, the newest model will be used. 
#### To run tests model: 
``` shell
uv run pytest
```