# 🐝 KubSmart: RL-Driven Kubernetes Resource Optimization

> **Project Documentation**  
> - 📄 [Dissertation: “Optimizing Memory Allocation for Pods in Kubernetes…”](./Kubsmart-Dissertation-Vishal-Khushlani-220400556.pdf)  
> - ✍️ [Reflective Essay: MSc Project Reflection](./MSc-Project\ Reflective-Essay-vishal-khushlani-220400556.pdf)

This repository contains reinforcement learning (RL) models trained to optimize memory and CPU resource requests and limits for Kubernetes pods. The work is detailed in the dissertation and reflective essay linked above.

---

## 📦 Virtual Environment Setup

### Prerequisites
- Python 3.8+
- `venv` (or an alternative virtual environment tool)
- `pip`

### Create & Activate
```bash
python -m venv .venv
source .venv/bin/activate     # On Linux/macOS
.\.venv\Scripts\activate      # On Windows PowerShell
```

---

## 🔧 Install Dependencies

Make sure you’re in the root of the project (same directory as `requirements.txt`), then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the Models

Each model is implemented in a separate script. You can run any of them independently after the environment is set up.

| Model | Description                             | Run Command               |
|-------|-----------------------------------------|---------------------------|
| PPO   | Proximal Policy Optimization            | `python rl_model_ppo.py`  |
| DQN   | Deep Q-Learning                         | `python rl_model_dqn.py`  |
| A2C   | Advantage Actor Critic                  | `python rl_model_a2c.py`  |
| SAC   | Soft Actor Critic (off-policy)          | `python rl_model_sa2c.py` |

---

## 📁 Repository Structure

```
.
├── rl_model_ppo.py             # PPO model training script
├── rl_model_dqn.py             # DQN model training script
├── rl_model_a2c.py             # A2C model training script
├── rl_model_sa2c.py            # SAC model training script
├── kubernetes_environment_rl.py # Custom Gym-compatible K8s environment
├── converter.py                # Resource string conversion utils
├── requirements.txt            # Python package dependencies
├── Kubsmart-Dissertation-*.pdf # Final project report
├── MSc-Project Reflective-*.pdf# Reflective essay
└── README.md                   # This file
```

---

## 🐞 Troubleshooting

- ✅ Ensure all pods are in the `Running` state in your Kubernetes cluster before starting any training.
- 🔑 Confirm your Kubernetes config is located at `~/.kube/config` or update the path in the environment code.
- 📦 If you run into dependency errors, try reinstalling with:
  ```bash
  pip install --force-reinstall -r requirements.txt
  ```

---

## 🧠 Learn More

To understand how each model works and how Kubernetes metrics are collected and applied, read:

- [Dissertation PDF](./Kubsmart-Dissertation-Vishal-Khushlani-220400556.pdf)
- [Reflective Essay](./MSc-Project\ Reflective-Essay-vishal-khushlani-220400556.pdf)

These documents include design decisions, experiments, analysis, and conclusions.

---

## 🙌 Acknowledgements

This project was developed as part of the MSc Computer Science program at Queen Mary University of London.

---

*Happy optimizing! 🚀*
