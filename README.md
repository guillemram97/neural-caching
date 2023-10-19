# Cache for LLM calls
## Setting up environment
```
conda create -n cache python=3.8
conda activate cache
conda install pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Usage
Export the arguments you want to change with respect to default values, following `scripts/run.sh`. For instance, 
```
export TASK_NAME=isear
export BUDGET=1500
```
We use [Neptune](http://neptune.ai/) for logging. We recommend setting file `scripts/cluster.sh` with the following arguments:
```
export DATA_PATH=<YOUR_DATA_PATH>
export PROJECT_NAME_NEPTUNE=<YOUR_NEPTUNE_PROJECT_NAME>
export API_TOKEN_NEPTUNE=<YOUR_NEPTUNE_API_TOKEN>
```
Then, run the code:
```
bash scripts/run.sh 
```
