# Stock Trading Market OpenAI Gym Environment with Deep Reinforcement Learning using Keras

## Overview

This project provides a general environment for stock market trading simulation using [OpenAI Gym](https://gym.openai.com/). 
Training data is a close price of each day, which is downloaded from Google Finance, but you can apply any data if you want.
Also, it contains simple Deep Q-learning and Policy Gradient from [Karpathy's post](http://karpathy.github.io/2016/05/31/rl/).

In fact, the purpose of this project is not only providing a best RL solution for stock trading, but also building a general open environment for further research.  
**So, please, manipulate the model architecture and features to get your own better solution.**

## Requirements

- Python2.7 or higher
- Numpy
- HDF5
- Keras with Beckend (Theano or/and Tensorflow)
- OpenAI Gym

## Usage

Note that the most sample training data in this repo is Korean stock. 
You may need to re-download your own training data to fit your purpose.

After meet those requirements in above, you can begin the training both algorithms, Deep Q-learning and Policy Gradient.

Train Deep Q-learning:

    $ python market_dqn.py <list filename> [model filename]

Train Policy Gradient:

	$ python market_pg.py <list filename> [model filename]

For example, you can do like this:

	$ python market_pg.py ./kospi_10.csv pg.h5

Aware that the provided neural network architecture in this repo is too small to learn.
So, it may under-fitting if you try to learn every stock data.
It just fitted for 10 to 100 stock data for a few years. (I checked!!)  
Thus you need to re-design your own architecture and   
**let me know if you have better one!**

Below is training curve for Top-10 KOSPI stock datas for 4 years using Policy Gradient.  
![Training Curve](./pg_over_top_10.png)

## To do
- Test environment to check overfitting.
- Elaborate the PG's train interface.

## Reference

[1] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[2] [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)  
[3] [KEras Reinforcement Learning gYM agents, KeRLym](https://github.com/osh/kerlym)  
[4] [Keras plays catch, a single file Reinforcement Learning example](http://edersantana.github.io/articles/keras_rl/)
