# Fully Parameterized Quantile Function (FQF) and Extensions

PyTorch implementation of the state-of-the-art distributional reinforcement learning algorithm Fully Parameterized Quantile Function (FQF).
Implementation includes also [DQN extensions](https://arxiv.org/abs/1710.02298) with which FQF represents the most powerful Rainbow version. 

For details on the algorithm check the [article on medium](https://medium.com/@sebastian.dittert3692/distributional-reinforcement-learning-part-2-iqn-and-fqf-567fbc7a04d7)

Extension included:
- **P**rioritized **E**xperience **R**eplay Buffer (PER)
- Noisy Layer for exploration
- N-step Bootstrapping
- Dueling Version
- [Munchausen RL](https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de)

#### Dependencies
Trained and tested on:
<pre>
Python 3.5.6 
PyTorch 1.4.0  
Numpy 1.15.2 
gym 0.10.11 
</pre>

## Train:

With the script version it is possible to train on simple environments like CartPole-v0 and LunarLander-v2 or on Atari games with image inputs!

To run the script version execute in your command line:
`python run.py -info fqf_run1`

To run the script version on the Atari game Pong:
`python run.py -env PongNoFrameskip-v4 -info fqf_pong1`

#### Hyperparameter
To see the options:
`python run.py -h`

    -agent, choices=["iqn","fqf+per","noisy_fqf","noisy_fqf+per","dueling","dueling+per", "noisy_dueling","noisy_dueling+per"], Specify which type of FQF agent you want to train, default is FQF - baseline!
    -env,  Name of the Environment, default = CartPole-v0
    -frames, Number of frames to train, default = 60000
    -eval_every, Evaluate every x frames, default = 1000
    -eval_runs, Number of evaluation runs, default = 5"
    -seed, Random seed to replicate training runs, default = 1
    -N, Number of quantiles, default = 32
    -ec, --entropy_coeff, Entropy coefficient, default = 0.001
    -bs, --batch_size, Batch size for updating the DQN, default = 8
    -layer_size, Size of the hidden layer, default=512
    -n_step, Multistep IQN, default = 1
    -m, --memory_size, Replay memory size, default = 1e5
    -u, --update_every, Update the network every x steps, default = 1
    -munchausen,  choices=[0,1], Use Munchausen RL loss for training if set to 1 (True), default = 0
    -lr, Learning rate, default = 5e-4
    -g, --gamma, Discount factor gamma, default = 0.99
    -t, --tau, Soft update parameter tat, default = 1e-2
    -eps_frames, Linear annealed frames for Epsilon, default = 5000
    -min_eps, Final epsilon greedy value, default = 0.025
    -info, Name of the training run
    -save_model, choices=[0,1]  Specify if the trained network shall be saved or not, default is 0 - not saved!

### Observe training results
  `tensorboard --logdir=runs`
  
  
## Results

#### CartPole Results
![alttext](/imgs/FQF_CP_Extensions_.png)

#### LunarLander Results
200000 Frames (~54 min), eps_frames: 20000, eval_every: 5000
![alttext](/imgs/FQF_IQN_LL_.png)

comparison 

## Help and issues:
Im open for feedback, found bugs, improvements or anything. Just leave me a message or contact me.

## Paper and References:

- [FQF](https://arxiv.org/pdf/1911.02140.pdf)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Noisy layer](https://arxiv.org/pdf/1706.10295.pdf)
- [C51](https://arxiv.org/pdf/1707.06887.pdf)
- [PER](https://arxiv.org/pdf/1511.05952.pdf)

Big thank you also to Toshiki Watanabe who helped me with the implementation and where I have the training routine for the fraction proposal network from! His [Repo](https://github.com/ku2482/fqf-iqn-qrdqn.pytorch)


## Author
- Sebastian Dittert

**Feel free to use this code for your own projects or research.**
For citation:
```
@misc{FQF and Extensions,
  author = {Dittert, Sebastian},
  title = {Fully Parameterized Quantile Function (FQF) and Extensions},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BY571/FQF-and-Extensions}},
}

