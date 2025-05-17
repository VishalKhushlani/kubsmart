# Virtual environment setup for running the models

## Prerequisites
* venv
* pip

# Setup

## Install all the required dependencies using requirements.txt 
```bash
pip install -r requirements.txt
```

## Run the ppo model
```bash
python rl_model_ppo.py
```

## Run the dqn model
```bash
python rl_model_dqn.py
```

## Run the dqn model
```bash
python rl_model_a2c.py
```

## Run the dqn model
```bash
python rl_model_sa2c.py
```

## Troubleshooting
* Ensure that all the pods are running in kubernetes before running the models
* Ensure that all the dependencies are installed correctly before running the models