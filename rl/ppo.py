"""
PPO Agent Module - Backward Compatibility Layer

This module re-exports PPO and ActorCriticNetwork from their new locations.
New code should import directly from:
  - agents.ppo.PPO
  - models.actor_critic.ActorCriticNetwork

Legacy imports from this module are still supported:
  from rl.ppo import PPO, ActorCriticNetwork
"""

from agents.ppo import PPO
from models.actor_critic import ActorCriticNetwork

__all__ = ['PPO', 'ActorCriticNetwork']
