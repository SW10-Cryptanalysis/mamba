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
#### To run tests model: 
``` shell
uv run pytest
```