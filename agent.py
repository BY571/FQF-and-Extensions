import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import networks
from networks import *
from ReplayBuffers import *



class FQF_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 network,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 Munchausen,
                 N,
                 entropy_coeff,
                 worker,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            entropy_coeff (float): entropy coefficient
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.network = network
        self.seed = random.seed(seed)
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.worker = worker
        self.UPDATE_EVERY = worker
        self.t_step = 0
        self.BATCH_SIZE = BATCH_SIZE*worker
        self.Q_updates = 0
        self.n_step = n_step
        self.entropy_coeff = entropy_coeff
        self.N = N
        # munchausen params
        self.munchausen = Munchausen
        self.entropy_tau = 0.03 #0.03
        self.entropy_tau_coeff = 1e-2
        self.lo = -1
        self.alpha = 0.9
    
        if "noisy" in self.network:
            noisy = True
        else:
            noisy = False
        
        if "duel" in self.network:
            duel = True
        else:
            duel = False

        # FQF-Network
        self.qnetwork_local = QVN(state_size, action_size,layer_size, n_step, device, seed, dueling=duel, noisy=noisy, N=N).to(device)
        self.qnetwork_target = QVN(state_size, action_size,layer_size, n_step,device, seed, dueling=duel, noisy=noisy, N=N).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        state_dim = len(self.state_size)
        if state_dim == 3:
            fpn_layer_size = 3136
        else:
            fpn_layer_size = layer_size
        self.FPN = FPN(fpn_layer_size, seed, N, device).to(device)
        print(self.FPN)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=LR*0.000001, alpha=0.95, eps=0.00001)
        
        # Replay memory
        if "per" in self.network:
            self.per = 1
            self.memory = PrioritizedReplay(BUFFER_SIZE, self.BATCH_SIZE, seed=seed, gamma=self.GAMMA, n_step=n_step, parallel_env=self.worker)
        else:
            self.per = 0
            self.memory = ReplayBuffer(BUFFER_SIZE, self.BATCH_SIZE, self.device, seed, self.GAMMA, n_step, self.worker)
        print("Using PER: {}".format(self.per))

    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                if not self.per:
                    loss, entropy = self.learn(experiences)
                else:
                    loss, entropy = self.learn_per(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)
                writer.add_scalar("Entropy", entropy, self.Q_updates)

                
    def act(self, state, eps=0., eval=False):
        """Returns actions for given state as per current policy"""
        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy actioLinearn if random number is higher than epsilon or noisy network is used!
            state = torch.from_numpy(state).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                embedding = self.qnetwork_local.forward(state)
                taus, taus_, _ = self.FPN(embedding)
                F_Z = self.qnetwork_local.get_quantiles(state, taus_, embedding)
                action_values = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z).sum(1)
                #assert action_values.shape == (1, self.action_size)
                
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            if eval:
                action = random.choices(np.arange(self.action_size), k=1)
            else:
                action = random.choices(np.arange(self.action_size), k=self.worker)
            return action


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())

        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
        assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
        
        # calc fractional loss 
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss
        # Calculate Q_targets without munchausen 
        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
        # Calculate Q_targets with munchausen
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.GAMMA**self.n_step * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1) 
        loss = loss.mean()
        

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy(), entropy.mean().detach().cpu().numpy()          



    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())
        
        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
        assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
        # calc fractional loss
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus, weights)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss

        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.GAMMA**self.n_step * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target


        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights
        loss = loss.mean()
        

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update priorities
        td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test 
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))
        return loss.detach().cpu().numpy(), entropy.mean().detach().cpu().numpy()    

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    
def calc_fraction_loss(FZ_,FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    assert not gradients.requires_grad
    if not weights == None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1)*weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss
    