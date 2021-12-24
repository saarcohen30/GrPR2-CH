from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy, DiscreteConditionalPolicy
import time

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, messg_dim, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol, messg_dim,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol, 
                                            messg_dim,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs_messg, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        result = self.policy(obs_messg, sample=explore)
        return result

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        
class AttentionREGMAAgent(object):
    """
    General class for REGMA Attention agents (opponent policy, policy, target opponent policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, messg_dim, action_dim, agent_num, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.opponent_policy = DiscretePolicy(num_in_pol, num_out_pol * (agent_num - 1), messg_dim,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
                                     
        self.policy = DiscreteConditionalPolicy(self.opponent_policy, 
                                     num_in_pol + action_dim * (agent_num - 1), num_out_pol, messg_dim,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)

        self.target_opponent_policy = DiscretePolicy(num_in_pol, num_out_pol * (agent_num - 1), messg_dim,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
                                     
        self.target_policy = DiscreteConditionalPolicy(self.target_opponent_policy, num_in_pol + action_dim * (agent_num - 1),
                                            num_out_pol, 
                                            messg_dim,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)
        
        hard_update(self.target_opponent_policy, self.opponent_policy)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.opponent_policy_optimizer = Adam(self.opponent_policy.parameters(), lr=lr)

    def step(self, obs_messg, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        result, _ = self.policy(obs_messg, sample=explore)
        return result

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.opponent_policy.load_state_dict(params['opponent_policy'])
        self.target_opponent_policy.load_state_dict(params['target_opponent_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
