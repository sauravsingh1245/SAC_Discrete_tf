import os
import tensorflow as tf
import numpy as np

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
        ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    """Applies adjustment to mean, pi and log prob.

    This formula is a little bit magic. To get an understanding of where it
    comes from, check out the original SAC paper (arXiv 1801.01290) and look
    in appendix C. This is a more numerically-stable equivalent to Eq 21.
    Try deriving it yourself as a (very difficult) exercise. :)
    """
    logp_pi -= tf.reduce_sum(
        2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation, kernel_regularizer = tf.keras.regularizers.L2(0.01))
        for size in hidden_sizes
    ], name)


def layer_norm_mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_sizes[0]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation(tf.nn.tanh),
        mlp(hidden_sizes[1:], activation)
    ], name)


def mlp_actor_critic(obs_dim,
                     act_dim,
                     hidden_sizes=(256, 256),
                     activation=tf.nn.relu):
    """Creates actor and critic tf.keras.Model-s."""

    obs_input = tf.keras.Input(shape=(obs_dim,))

    # Make the actor.
    body_actor = tf.keras.Sequential([
        mlp(hidden_sizes, activation, name='actor'),
        tf.keras.layers.Dense(act_dim, activation=tf.nn.softmax),
        #tf.keras.layers.Reshape([])  # Very important to squeeze values!
    ])
    actor = tf.keras.Model(inputs=[obs_input],
                            outputs=body_actor(obs_input))

    # Make the critic.
    body_critic = tf.keras.Sequential([
        mlp(hidden_sizes, activation, name='critic'),
        tf.keras.layers.Dense(act_dim),
        #tf.keras.layers.Reshape([])  # Very important to squeeze values!
    ])
    critic = tf.keras.Model(inputs=[obs_input],
                            outputs=body_critic(obs_input))

    return actor, critic