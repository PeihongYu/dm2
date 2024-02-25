# from components.episode_buffer import EpisodeBatch
# from torch.optim import RMSprop
import os.path as osp
import sys
import copy
import numpy as np
import torch as th
import torch.nn as nn
from components.episode_buffer import EpisodeBatch
from controllers.dcntrl_controller import DcntrlMAC
from utils.mappo_utils.util import get_grad_norm, huber_loss, mse_loss
from utils.mappo_utils.valuenorm import ValueNorm
from utils.mappo_utils.separated_buffer import SeparatedReplayBuffer
from utils.mappo_utils.util import check, update_linear_schedule
import utils.mappo_utils.separated_buffer as sbuffer
from learners.gail_learner2 import GailDiscriminator2

np.set_printoptions(threshold=sys.maxsize)

class IPPOLearner2:
    def __init__(self, mac: DcntrlMAC, scheme, logger, args, obs_info=None):
        ''' 
        obs_info: information about the observation dimensions
        '''
        if args.use_cuda:
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")
        self.args = args
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.use_linear_lr_decay = args.use_linear_lr_decay
        self.optim_eps = args.optim_eps
        self.weight_decay = args.weight_decay
        self.tpdv = dict(dtype=th.float32, device=self.device)
        self.n_agents =  args.n_agents
        self.t_max = args.t_max
        self.episode_limit = args.episode_limit
        self.batch_size_run = args.batch_size_run
        self.batch_size = args.batch_size

        self.mac = mac
        self.state_shape = self.mac.input_scheme['state']["vshape"]
        self.obs_shape = self.mac.input_scheme['obs']["vshape"]
        self.n_actions = args.n_actions
        self.logger = logger
        self.log_prefix = "ippo-gail_" if args.rew_type in  ["gail", "mixed"] else "ippo_"
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.buffers = [SeparatedReplayBuffer(args,  self.obs_shape, self.state_shape)
                        for _ in range(self.n_agents)]

        self.clip_param = self.args.clip_param
        self.ppo_epoch = self.args.ppo_epoch
        self.num_mini_batch = self.args.num_mini_batch
        self.data_chunk_length = self.args.data_chunk_length
        self.value_loss_coef = self.args.value_loss_coef
        self.entropy_coef = self.args.entropy_coef
        self.max_grad_norm = self.args.max_grad_norm       
        self.huber_delta = self.args.huber_delta
        self.gamma = args.gamma

        self._use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        self._use_recurrent_policy = self.args.use_recurrent_policy

        self._use_max_grad_norm = self.args.use_max_grad_norm
        self._use_clipped_value_loss = self.args.use_clipped_value_loss
        self._use_huber_loss = self.args.use_huber_loss

        self._use_value_active_masks = self.args.use_value_active_masks
        self._use_policy_active_masks = self.args.use_policy_active_masks
        
        self.actor_params = mac.parameters()
        self.critic_params = mac.critic_parameters()
        self.actor_optimizers = []
        self.critic_optimizers = []
        for n in range(self.n_agents):
            self.actor_optimizers.append(th.optim.Adam(self.actor_params[n],
                                                 lr=self.lr, eps=self.optim_eps,
                                                 weight_decay=self.weight_decay))
            self.critic_optimizers.append(th.optim.Adam(self.critic_params[n],
                                              lr=self.critic_lr,
                                              eps=self.optim_eps,
                                              weight_decay=self.weight_decay))

        self.no_op_tensor = th.zeros((self.n_actions,), dtype=th.float32, device=self.device)
        self.no_op_tensor[0] = 1

        # setup gail 
        self.rew_type = args.rew_type
        self.update_gail = args.update_gail

        self.gail_obs_discr = args.gail_obs_discr
        self.save_gail_data = args.save_agent_batches
        self.gail_load_models = args.gail_load_models
        self.gail_sil = args.gail_sil
        # scenarios: ippo w/env reward, ippo w/env reward + saving data for gail, 
        # ippo training w/gail reward
        if self.rew_type in  ["gail", "mixed"] or self.gail_load_models:
            if self.args.gail_exp_use_single_seed: # use same expert data for all agents
                self.gail_data_paths = [self.args.gail_data_paths[0] for i in range(self.n_agents)]
            else:
                assert len(self.args.gail_data_paths) == self.n_agents
                self.gail_data_paths = self.args.gail_data_paths

            self.gail_batch_size = args.gail_batch_size
            self.gail_epoch = args.gail_epoch
            self.gail_rew_coef = args.gail_rew_coef

            gail_agent_paths = [osp.join(self.gail_data_paths[i], f"agent_{i}") for i in range(self.n_agents)]        

        if self.save_gail_data or self.gail_sil:
            gail_agent_paths = [None for i in range(self.n_agents)]


        self.gails = None
        if self.rew_type in ["gail", "mixed"] or self.save_gail_data or self.gail_load_models:
            if self.args.gail_state_discrim:
                gail_input_dim = self.obs_shape
            else: 
                gail_input_dim = self.obs_shape + 1

            self.gails = [GailDiscriminator2(self.args, 
                                            input_dim=gail_input_dim, 
                                            hidden_dim=args.mlp_hidden_dim, 
                                            device=self.device, 
                                            max_buffer_eps=self.args.save_agent_batchsize if self.save_gail_data else self.args.gail_buffer_size,
                                            epath=gail_agent_paths[i],
                                            agent_idx=i, 
                                            obs_info=obs_info) for i in range(self.n_agents)
                          ]
        
        if self.gail_load_models:
            self.load_gails(args.gail_model_paths, load_same_model=args.gail_use_same_model, load_optimisers=False)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        for n in range(self.n_agents):
            update_linear_schedule(self.actor_optimizers[n], episode, episodes, self.lr)
            update_linear_schedule(self.critic_optimizers[n], episode, episodes, self.critic_lr)

    def insert_episode_batch(self, ep_batch: EpisodeBatch):
        for i, buffer in enumerate(self.buffers):
            # shape: (1, ts, n_agents, n_feats)
            state = ep_batch["state"]
            obses = ep_batch["obs"][:, :, i, :]
            rewards = ep_batch["reward"]
            rnn_states_actors = ep_batch["rnn_states_actors"][:, :, i, :]
            rnn_states_critics = ep_batch["rnn_states_critics"][:, :, i, :]
            actions = ep_batch["actions"][:, :, i, :]
            if self.rew_type in  ["gail", "mixed"] or self.save_gail_data:
                self.gails[i].add_agent_data(obses, actions)
                if self.gail_sil:
                    self.gails[i].add_sil_expert_data(obses, actions, rewards)

            actions_onehot = ep_batch["actions_onehot"][:, :, i, :]
            available_actions = ep_batch["avail_actions"][:, :, i, :]

            max_ep_t = ep_batch.max_t_filled()
            max_agent_t = th.sum((available_actions[:, :max_ep_t] != self.no_op_tensor)[:, :, 0]) # add 1 to be consistent with max_ep_t

            # the terminated mask denotes whether the agent has died
            assert max_agent_t <= max_ep_t, f"max_agent_t={max_agent_t}, max_ep_t={max_ep_t}"
            terminated_mask = th.ones((1, buffer.episode_limit + 1, 1), dtype=th.float32, device=self.device) # should indicate when ep is done
            terminated_mask[min(max_agent_t, max_ep_t):] = 0.

            buffer.insert(state, obses, 
                          rnn_states_actors, rnn_states_critics, 
                          actions, actions_onehot,
                          rewards, 
                          terminated_mask,
                          available_actions=available_actions)

    def insert_episode_batch_gail_only(self, ep_batch:EpisodeBatch):
        # called in run_ippo.py
        for i, gail in enumerate(self.gails):
            # shape: (1, ts, n_agents, n_feats)
            obses = ep_batch["obs"][:, :, i, :]
            actions = ep_batch["actions"][:, :, i, :]
            gail.add_agent_data(obses, actions)


    def cal_value_loss(self, values, value_preds_batch, return_batch, terminated_batch):
        """
        Calculate value function loss.
        :param values: (th.Tensor) value function predictions.
        :param value_preds_batch: (th.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (th.Tensor) reward to go returns.
        :param terminated_batch: (th.Tensor) denotes if episode has terminated or if agent has died.
        :return value_loss: (th.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                     self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = th.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            # TODO: overload terminated masks with agents dying
            value_loss = (value_loss * terminated_batch).sum() / terminated_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, agent_id, obs_batch, rnn_states_actor_batch,
                   rnn_states_critic_batch, actions_batch, value_preds_batch,
                   return_batch, terminated_batch, old_action_log_probs_batch,
                   adv_targ, available_actions_batch, update_actor=True):
        """
        Update actor and critic networks for agent agent_id
        :update_actor: (bool) whether to update actor network.
        :return value_loss: (th.Tensor) value function loss.
        :return critic_grad_norm: (th.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (th.Tensor) actor(policy) loss value.
        :return dist_entropy: (th.Tensor) action entropies.
        :return actor_grad_norm: (th.Tensor) gradient norm from actor update.
        :return imp_weights: (th.Tensor) importance sampling weights.
        """
        action_log_probs, dist_entropy = self.mac.eval_action_ippo(agent_id, 
                                                                   obs_batch,
                                                                   actions_batch, 
                                                                   available_actions_batch, 
                                                                   rnn_states_actor_batch)

        values = self.mac.get_value_ippo(agent_id, 
                                         obs_batch,
                                         rnn_states_critic_batch)
        # actor update
        imp_weights = th.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-th.sum(th.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * terminated_batch).sum() / terminated_batch.sum()
        else:
            policy_action_loss = -th.sum(th.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizers[agent_id].zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_params[agent_id], self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor_params[agent_id])

        self.actor_optimizers[agent_id].step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, terminated_batch)

        self.critic_optimizers[agent_id].zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic_params[agent_id], self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.critic_params[agent_id])

        self.critic_optimizers[agent_id].step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights


    def gail_update(self, agent_id:int, train_info: dict):
        n_gail_updates = self.n_agents * self.gail_epoch
        for _ in range(self.gail_epoch):
            expert_batch = self.gails[agent_id].expert_storage.get_random_batch(self.gail_batch_size) # change this function to partition current buffer into minibatches
            agent_batch = self.gails[agent_id].agent_storage.get_random_batch(self.gail_batch_size) # change this function to partition current buffer into minibatches
            # gail_total_loss, gail_grad_norm, gail_grad_pen, gail_policy_discr_pred, gail_expert_discr_pred = self.gails[agent_id].update(agent_batch, expert_batch)
            gail_total_loss, gail_grad_norm, gail_grad_pen, gail_policy_discr_pred, gail_expert_discr_pred, gail_dyn_loss = self.gails[agent_id].update(agent_batch, expert_batch)

            train_info['gail_total_loss'] += gail_total_loss / n_gail_updates
            train_info['gail_grad_norm'] += gail_grad_norm / n_gail_updates
            train_info['gail_grad_pen'] += gail_grad_pen / n_gail_updates
            train_info['gail_policy_discr_pred'] += gail_policy_discr_pred / n_gail_updates
            train_info['gail_expert_discr_pred'] += gail_expert_discr_pred / n_gail_updates
            train_info['gail_dyn_loss'] += gail_dyn_loss / n_gail_updates
        return train_info


    def train(self, t_env):
        """
        Perform a training update over each agent using minibatch GD
        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if not self.buffers[0].can_sample():
            return
        if self.update_gail:
            for agent_id in range(self.n_agents):
                if not self.gails[agent_id].can_sample():
                    return
        print("TRAINING IPPO")
        if self.use_linear_lr_decay:
            self.lr_decay(t_env, self.t_max)

        num_updates = self.ppo_epoch * self.num_mini_batch * self.n_agents
        train_info = {
            'value_loss': 0.,
            'policy_loss': 0.,
            'dist_entropy': 0.,
            'actor_grad_norm': 0.,
            'critic_grad_norm': 0.,
            'ratio': 0.
        }
        if self.rew_type in  ["gail", "mixed"] or self.gail_obs_discr:
            train_info = {**train_info, 
            'gail_pred_rew': 0.,
            'gail_total_loss': 0.,
            'gail_grad_norm': 0.,
            'gail_grad_pen': 0.,
            'gail_policy_discr_pred': 0.,
            'gail_expert_discr_pred': 0.,
            'gail_dyn_loss': 0.,
            'mixed_rew': 0.
            }

        for agent_id in range(self.n_agents):
            batch = self.buffers[agent_id].get_batch()

            if self.update_gail:
                train_info = self.gail_update(agent_id, train_info)

            pred_rew = None
            if self.rew_type in  ["gail", "mixed"] or self.gail_obs_discr:
                # predict reward 
                pred_rew, weights = self.gails[agent_id].predict_reward(obs=batch['obs'], actions=None if self.args.gail_state_discrim else batch["actions"])
                train_info['gail_pred_rew'] += th.mean(pred_rew[:, :-1]).item() / self.n_agents
            # compute reward for ppo learner
            if self.rew_type == "gail":
                rewards = pred_rew[:, :-1] * weights[:, :-1]
            elif self.rew_type == "mixed":
                rewards = pred_rew[:, :-1] * weights[:, :-1] * self.gail_rew_coef + batch["reward"][:, :-1]
                train_info['mixed_rew'] += th.mean(rewards).item() / self.n_agents                    
            else:
                rewards = batch["reward"][:, :-1]  # rewards are length T+1

            actions = batch["actions"][:, :-1]
            actions_onehot_all = batch["actions_onehot"]
            avail_actions = batch["available_actions"][:, :-1]

            terminated_all = batch["terminated_masks"]
            terminated = batch["terminated_masks"][:, :-1]

            rnn_state_actor = batch["rnn_states_actor"][:, :-1]
            rnn_state_critic_all = batch["rnn_states_critic"]
            rnn_state_critic = batch["rnn_states_critic"][:, :-1]

            obs_all = batch["obs"]
            obs_all = self.mac._build_inputs_ippo(agent_id, obs_all, actions_onehot_all, 
                discr_signal=pred_rew if self.gail_obs_discr else None
                )
            obs = obs_all[:, :-1]

            # compute advantage
            returns = self.compute_returns(agent_id, obs_all, rewards, terminated_all, rnn_state_critic_all).clone().detach()
            current_values = self.mac.get_value_ippo(agent_id, obs, rnn_state_critic).clone().detach()
            advantages = returns - current_values

            # advantage normalization
            advantages_copy = advantages.clone().detach()
            advantages_copy[terminated == 0.0] = 0.
            std_advantages, mean_advantages = th.std_mean(advantages_copy)
            advantages = (advantages_copy - mean_advantages) / (std_advantages + 1e-5)

            action_log_probs, _ = self.mac.eval_action_ippo(agent_id, obs, actions, avail_actions, rnn_state_actor)
            action_log_probs = action_log_probs.clone().detach()

            for _ in range(self.ppo_epoch): # default: ppo_epoch=15

                data_generator = self.generate_data(obs, rnn_state_actor, rnn_state_critic, actions, returns,
                                                    terminated, action_log_probs, advantages, avail_actions,
                                                    current_values, self.num_mini_batch)

                for sample in data_generator:
                    obs_batch, rnn_states_actor_batch, \
                    rnn_states_critic_batch, actions_batch, value_preds_batch, \
                    return_batch, terminated_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch = sample

                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.ppo_update(agent_id, obs_batch, 
                                          rnn_states_actor_batch, rnn_states_critic_batch, 
                                          actions_batch,
                                          value_preds_batch, return_batch, 
                                          terminated_batch, 
                                          old_action_log_probs_batch,
                                          adv_targ, available_actions_batch)

                    train_info['value_loss'] += value_loss.item()/num_updates
                    train_info['policy_loss'] += policy_loss.item()/num_updates
                    train_info['dist_entropy'] += dist_entropy.item()/num_updates
                    train_info['actor_grad_norm'] += actor_grad_norm/num_updates
                    train_info['critic_grad_norm'] += critic_grad_norm/num_updates
                    train_info['ratio'] += imp_weights.mean().item()/num_updates

            self.buffers[agent_id].clear_buffer()
            if self.rew_type in  ["gail", "mixed"]:
                self.gails[agent_id].flush()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for k, v in train_info.items():
                self.logger.log_stat(self.log_prefix + k, v, t_env)

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        for i in range(self.n_agents):
            th.save(self.actor_optimizers[i].state_dict(), "{}/actor_{}_opt.th".format(path, i))
            th.save(self.critic_optimizers[i].state_dict(), "{}/critic_{}_opt.th".format(path, i))
            if self.rew_type in  ["gail", "mixed"]:
                th.save(self.gails[i].trunk.state_dict(), "{}/discriminator_{}.th".format(path, i))
                th.save(self.gails[i].optimizer.state_dict(), "{}/discriminator_{}_opt.th".format(path, i))
                th.save(self.gails[i].dyn_discriminator.state_dict(), "{}/dyn_discriminator_{}.th".format(path, i))
                th.save(self.gails[i].dyn_optimizer.state_dict(), "{}/dyn_discriminator_{}_opt.th".format(path, i))

    def save_traj_data(self, root_path):
        for i in range(self.n_agents):
            self.gails[i].save_agent_data("{}/agent_{}".format(root_path, i))
            self.gails[i].flush() 

    def load_models(self, paths:list, load_optimisers=False):
        '''If paths has multiple entries: Load ith agent model from ith path.
        Else, if paths has 1 entry, load all agent models from the path. '''
        self.mac.load_models(paths)

        # Not quite right but I don't want to save target networks or replay buffers
        if load_optimisers:
            if len(paths) == 1:
                path = copy.copy(paths[0])
                paths = [path for i in range(self.n_agents)]
            for i in range(self.n_agents):
                self.actor_optimizers[i].load_state_dict(th.load("{}/actor_{}_opt.th".format(paths[i], i),
                    map_location=lambda storage, loc: storage))
                self.critic_optimizers[i].load_state_dict(th.load("{}/critic_{}_opt.th".format(paths[i], i),
                    map_location=lambda storage, loc: storage))

    def load_gails(self, paths:list, load_same_model=False, load_optimisers=False):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]
        for i in range(self.n_agents):
            self.gails[i].load_model(paths[i], load_same_model=load_same_model, load_optimisers=load_optimisers)

    def compute_returns(self, agent_id, obs_all, rewards, terminated, rnn_state_critic_all):
        """
        Take as input the batch of obs, critic_rnn_state, rewards and masks to then compute the returns
        Shape: (n_eps, timesteps, feat_size)
        """
        value_preds = self.mac.get_value_ippo(agent_id, obs_all, rnn_state_critic_all)
        returns = []
        T = rewards.shape[1]
        if self._use_gae:
            gae = 0
            for step in reversed(range(T)):
                delta = rewards[:, step] + self.gamma * value_preds[:, step + 1] * terminated[:, step + 1] - \
                        value_preds[:, step]
                gae = delta + self.gamma * self.gae_lambda * terminated[:, step + 1] * gae
                returns.append(gae + value_preds[:, step])
        else:
            returns.append(value_preds[:, -1])
            for step in reversed(range(T)):
                returns.append(returns[-1] * self.gamma * terminated[:, step + 1] + rewards[:, step])
        returns = th.flip(th.cat(returns, axis=1), dims=[1]).unsqueeze(-1)
        return returns[:, :T]

    """ DATA GENERATION CODE FROM SEPARATED BUFFER """
    def generate_data(self, obs, rnn_states_actor, rnn_states_critic, actions, returns,
                      terminated, action_log_probs, advantages, available_actions,
                      value_preds,
                      num_mini_batch=None, mini_batch_size=None):
        """Generate minibatches of data from batch. Timesteps are scrambled."""
        episode_limit, batch_size = self.episode_limit, self.batch_size
        batch_size = batch_size * episode_limit # 1023 * 61

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                        "PPO requires the number of processes ({}) "
                        "* number of steps ({}) = {} "
                        "to be greater than or equal to the number of PPO mini batches ({})."
                        "".format(batch_size_run, episode_limit, batch_size_run * episode_limit,
                                  num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch # batch_size * 60 / 1
        rand = th.randperm(batch_size).numpy()

        sampler = [th.Tensor(rand[i * mini_batch_size:(i + 1) * mini_batch_size]).long().to(self.device) for i in range(num_mini_batch)]

        obs = obs.reshape(-1, *obs.shape[2:])
        rnn_states_actor = rnn_states_actor.reshape(-1, *rnn_states_actor.shape[2:])
        rnn_states_critic = rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:])
        actions = actions.reshape(-1, actions.shape[-1])

        if available_actions is not None:
            available_actions = available_actions[:-1].reshape(-1, available_actions.shape[-1])

        value_preds = value_preds.reshape(-1, 1) # Why are we cutting off value preds 1 before?? Leads to a size inconsistency
        returns = returns.reshape(-1, 1)
        terminated = terminated.reshape(-1, 1)
        action_log_probs = action_log_probs.reshape(-1, action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            rnn_states_actor_batch = rnn_states_actor[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None

            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            terminated_batch = terminated[indices]
            old_action_log_probs_batch = action_log_probs[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, terminated_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
