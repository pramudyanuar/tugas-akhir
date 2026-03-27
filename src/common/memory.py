"""Memory buffer for storing trajectories."""

import numpy as np
import torch


class Memory:
    """
    Storage untuk trajectories dalam episode.
    
    Store: states, actions, rewards, values, log_probs, masks, dones
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.action_masks = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, action_mask, done):
        """Tambah transition ke memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory setelah update."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.action_masks.clear()
        self.dones.clear()
    
    def get_batch(self):
        """Get batch data untuk training."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        action_masks = torch.FloatTensor(np.array(self.action_masks))
        dones = torch.FloatTensor(np.array(self.dones))
        
        return states, actions, rewards, values, log_probs, action_masks, dones
