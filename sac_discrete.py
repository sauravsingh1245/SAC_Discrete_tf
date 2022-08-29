import os
import random
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from RL_Agents.SAC_Discrete.buffer import ReplayBuffer
import RL_Agents.SAC_Discrete.networks as nw

class Agent():
    def __init__(self, obs_dim, act_dim, env_id='SAC-discrete', polyak=0.9, gamma=0.99, replay_max_size=10_000, 
                    layer1_size=32, layer2_size=32, batch_size=256, seed=0, lr=0.001, update_every=128, update_after=256, save_path=None):
        
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Experience buffer.
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1,
                                    size=replay_max_size)
        self.gamma = gamma
        self.env_id = env_id
        self.polyak = polyak
        self.batch_size = batch_size
        self.save_path = save_path
        self.update_after = update_after
        self.update_every = update_every
        self.update_counter = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim


        # Build actor and critics networks.
        self.actor, self.critic = nw.mlp_actor_critic(obs_dim, act_dim,
                     hidden_sizes=(layer1_size, layer2_size), activation=tf.nn.relu)
        
        self.critic1 = self.critic
        self.critic2 = tf.keras.models.clone_model(self.critic)
        
        self.target_critic1 = tf.keras.models.clone_model(self.critic)
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.target_critic2 = tf.keras.models.clone_model(self.critic)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables

        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        if type(lr) == list:
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr[0])
            self.optimizer_critic1 = tf.keras.optimizers.Adam(learning_rate=lr[0])
            self.optimizer_critic2 = tf.keras.optimizers.Adam(learning_rate=lr[0])
            self.optimizer_alpha = tf.keras.optimizers.Adam(learning_rate=lr[1])
        else:
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr)
            self.optimizer_critic1 = tf.keras.optimizers.Adam(learning_rate=lr)
            self.optimizer_critic2 = tf.keras.optimizers.Adam(learning_rate=lr)
            self.optimizer_alpha = tf.keras.optimizers.Adam(learning_rate=lr)

        # temperature variable to be learned, and its target entropy
        self.log_alpha = tf.Variable(tf.math.log(1.0))
        self.alpha = tf.math.exp(self.log_alpha)

        #self.log_alpha = tf.Variable(0.00001)
        #self.alpha = tf.math.exp(self.log_alpha)
        self.target_entropy = 0.98 * -np.log(1.0 / act_dim)


    def get_action(self, state, deterministic=False):
        action_probabilities = self.actor(tf.expand_dims(tf.convert_to_tensor(state), 0))[0]
        print(f'action probs: {action_probabilities}')
        if deterministic:
            return tf.math.argmax(action_probabilities).numpy()
        else:
            action_distribution = tfp.distributions.Categorical(probs=action_probabilities)
            return action_distribution.sample().numpy()

    @tf.function
    def get_action_info(self, state):
        #Given the state, produces the probability of the action, the log probability of the action
        action_probabilities = self.actor(state)
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = tf.cast(action_probabilities==0.0,tf.float32) * 1e-8
        log_action_probabilities = tf.math.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities


    @tf.function
    def learn_on_batch(self, obs1, obs2, acts, rews, done):
        # Calculate target Q values
        # Q_tar = r + gamma * (target_Q(s', a') - alpha * log pi(a'|s'))
        with tf.GradientTape(persistent=True) as g:
            # Calculate critic loss
            # Get actions and log probs of actions for next states.
            action_probabilities, log_action_probabilities = self.get_action_info(obs2)
            qf1_next_target = self.target_critic1([obs2])
            qf2_next_target = self.target_critic2([obs2])
            min_qf_next_target = action_probabilities * (tf.minimum(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = tf.reduce_sum(min_qf_next_target,1)
            next_q_value = tf.stop_gradient(rews + (1.0 - done) * self.gamma * min_qf_next_target)
            
            coords = tf.concat([tf.transpose([tf.range(self.batch_size)]),tf.transpose([tf.cast(tf.squeeze(acts), tf.int32)])],1)     
            qf1 = tf.gather_nd(self.critic1(obs1),indices=coords)
            qf2 = tf.gather_nd(self.critic2(obs1),indices=coords)
            
            qf1_loss = self.mse(qf1, next_q_value)
            qf2_loss = self.mse(qf2, next_q_value)
            
            # Calculate actor loss
            action_probabilities, log_action_probabilities = self.get_action_info(obs1)
            qf1_pi = self.critic1(obs1)
            qf2_pi = self.critic2(obs1)
            min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
            inside_term = self.alpha * log_action_probabilities - min_qf_pi
            policy_loss = tf.reduce_mean(tf.reduce_sum(action_probabilities * inside_term, axis=1))
            
            # Calculate entrophy tuning loss
            alpha_loss = -tf.reduce_mean(self.log_alpha * (log_action_probabilities + self.target_entropy))

        
        #tf.print(self.log_alpha, self.log_alpha*(log_action_probabilities[:3,:]+self.target_entropy), alpha_loss)
            
        # Compute gradients and do updates.
        actor_gradients = g.gradient(policy_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        critic1_gradients = g.gradient(qf1_loss, self.critic1.trainable_variables)
        self.optimizer_critic1.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
        critic2_gradients = g.gradient(qf2_loss, self.critic2.trainable_variables)
        self.optimizer_critic2.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

        alpha_gradients = g.gradient(alpha_loss, [self.log_alpha])
        self.optimizer_alpha.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        del g

        # Soft update for target critic networks
        # Polyak averaging for target variables.
        for v, target_v in zip(self.critic1.trainable_variables,
                            self.target_critic1.trainable_variables):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        for v, target_v in zip(self.critic2.trainable_variables,
                            self.target_critic2.trainable_variables):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)

        return dict(pi_loss=policy_loss,
                    q1_loss=qf1_loss,
                    q2_loss=qf2_loss,
                    q1=qf1,
                    q2=qf2,
                    logp_pi=log_action_probabilities)


    def learn(self):
        self.update_counter += 1
        if self.update_counter > self.update_every and self.replay_buffer.size > self.update_after:
            #print(f'Learning... update_counter: {self.update_counter}; buffer size: {self.replay_buffer.size}')
            self.update_counter = 0
            for _ in range(self.update_every):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                results = self.learn_on_batch(**batch)
                self.alpha = tf.math.exp(self.log_alpha)



    # Store experience to replay buffer.
    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, reward, new_state, done)


    def save_models(self, path=None):
        if path is not None:
            print('.... Saving models ....')
            tf.keras.models.save_model(self.actor, os.path.join(path, self.env_id+'_actor'))
            tf.keras.models.save_model(self.critic1, os.path.join(path, self.env_id+'_critic1'))
            tf.keras.models.save_model(self.critic2, os.path.join(path, self.env_id+'_critic2'))
            tf.keras.models.save_model(self.target_critic1, os.path.join(path, self.env_id+'_target_critic1'))
            tf.keras.models.save_model(self.target_critic2, os.path.join(path, self.env_id+'_target_critic2'))
        elif self.save_path is not None:
            print('.... Saving models ....')
            tf.keras.models.save_model(self.actor, os.path.join(self.save_path, self.env_id+'_actor'))
            tf.keras.models.save_model(self.critic1, os.path.join(self.save_path, self.env_id+'_critic1'))
            tf.keras.models.save_model(self.critic2, os.path.join(self.save_path, self.env_id+'_critic2'))
            tf.keras.models.save_model(self.target_critic1, os.path.join(self.save_path, self.env_id+'_target_critic1'))
            tf.keras.models.save_model(self.target_critic2, os.path.join(self.save_path, self.env_id+'_target_critic2'))
        else:
            print('.... Save path not specified ....')



    def load_models(self, path=None):
        if path is not None:
            try:
                print('.... Loading models ....')
                
                self.actor = tf.keras.models.load_model(os.path.join(path, self.env_id+'_actor'))
                self.critic1 = tf.keras.models.load_model(os.path.join(path, self.env_id+'_critic1'))
                self.critic2 = tf.keras.models.load_model(os.path.join(path, self.env_id+'_critic2'))
                self.target_critic1 = tf.keras.models.load_model(os.path.join(path, self.env_id+'_target_critic1'))
                self.target_critic2 = tf.keras.models.load_model(os.path.join(path, self.env_id+'_target_critic2'))
            except:
                print('.... Models not found ....')
        elif self.save_path is not None:
            try:
                print('.... Loading models ....')
                self.actor = tf.keras.models.load_model(os.path.join(self.save_path, self.env_id+'_actor'))
                self.critic1 = tf.keras.models.load_model(os.path.join(self.save_path, self.env_id+'_critic1'))
                self.critic2 = tf.keras.models.load_model(os.path.join(self.save_path, self.env_id+'_critic2'))
                self.target_critic1 = tf.keras.models.load_model(os.path.join(self.save_path, self.env_id+'_target_critic1'))
                self.target_critic2 = tf.keras.models.load_model(os.path.join(self.save_path, self.env_id+'_target_critic2'))
            except:
                print('.... Models not found ....')
        else:
            print('.... Load path not specified ....')


    def save_replay_memory(self, path=None):
        if path is not None:
            print('.... Saving replay buffer ....')
            with open(os.path.join(path, self.env_id+'_replay_memory.pickle'), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        elif self.save_path is not None:
            print('.... Saving replay buffer ....')
            with open(os.path.join(self.save_path, self.env_id+'_replay_memory.pickle'), 'wb') as f:
                pickle.dump(self.replay_buffer, f)
        else:
            print('.... Save path not specified ....')
            

    def load_replay_memory(self, path=None):
        if path is not None:
            try:
                print('.... Loading Buffer ....')
                with open(os.path.join(path, self.env_id+'_replay_memory.pickle'), 'rb') as f:
                    self.replay_buffer = pickle.load(f)
            except:
                print('.... Replay Buffer not found ....')
        elif self.save_path is not None:
            try:
                print('.... Loading Buffer ....')
                with open(os.path.join(self.save_path, self.env_id+'_replay_memory.pickle'), 'rb') as f:
                    self.replay_buffer = pickle.load(f)
            except:
                print('.... Replay Buffer not found ....')
        else:
            print('.... Load path not specified ....')