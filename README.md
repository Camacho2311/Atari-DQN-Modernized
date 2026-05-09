# Atari-DQN-Modernized

This project modernizes an Atari Deep Q-Network (DQN) implementation using PyTorch and Gymnasium for reinforcement learning experiments on Atari environments such as Pong, Breakout, and Boxing.

## Features

* Gymnasium/ALE environment compatibility
* Apple Silicon (MPS) compatibility
* Updated Atari environment naming conventions
* Live gameplay evaluation support
* Modern Python 3.13 compatibility

## Supported Models

* DQN
* Double DQN (DDQN)
* Dueling DQN

## Supported Games

* Pong
* Breakout
* Boxing

## Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision
pip install "gymnasium[atari]"
pip install opencv-python matplotlib "imageio[ffmpeg]"
```

## Run Training

Example:

```bash
python3 main.py --env-name pong --model dueldqn --ddqn --epoch 501 --eval-cycle 50
```

## Live Gameplay Demo

```bash
python3 play_live.py
```

## Modifications Made

This repository was updated to support:

* Gymnasium Atari API changes
* ALE environment registration
* Apple Silicon GPU compatibility
* Updated tensor dtype handling
* Modern Python compatibility

## Original Repository Credit

Original repository:
[https://github.com/iewug/Atari-DQN](https://github.com/iewug/Atari-DQN)
