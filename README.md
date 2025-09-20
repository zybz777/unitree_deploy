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
- Install unitree_sdk2_python
```bash
# install cyclonedds
cd ..
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
git checkout 0.10.2
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
cd ..
export CYCLONEDDS_HOME="${pwd}/install"
# install unitree_sdk2_python
cd ..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```
## Usage
- Deploy policy to sim (↑↓←→ control robot)
```bash
source .venv/bin/activate
python ./deploy_sim/play.py
```
- Deploy policy to real
```bash
source .venv/bin/activate
python ./deploy_real/play.py --net=xxx
```