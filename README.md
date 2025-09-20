# Unitree Deploy

## Installation
- Install uv
```bash
# ubuntu
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc
```
- Clone this repo
```bash
git clone https://github.com/zybz777/unitree_deploy.git
cd unitree_deploy
uv sync
source .venv/bin/activate
```
- Install torch
```bash
# python 3.11 cuda12.8
uv pip install torch torchvision
```
- Install mujoco
```bash
uv pip install mujoco
```
## Usage
- Deploy policy to sim
```bash
source .venv/bin/activate
python ./deploy_sim/play.py
```
- Deploy policy to real
```bash
```