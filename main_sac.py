import os
import gym
import numpy as np
from sac_discrete import Agent
from utils import plot_learning_curve

#MAIN_DIR = 'C:\\Users\\ss3337\\Desktop\\saurav_data\\Continuous_SAC_tester\\RL_Agents\\sac_fixer'
MAIN_DIR = 'C:\\Users\\saura\\OneDrive\\Desktop\\Research\\ContinuousTaskSAC\\SAC_tf2_Discrete'
HIST_LEN = 100

if __name__ == '__main__':
    env_id = 'LunarLander-v2'
    #env_id = 'CartPole-v0'
    #env_id = 'MountainCar-v0'
    env = gym.make(env_id)
    agent = Agent(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.n, replay_max_size=50_000, lr=[0.005, 0.00003],
                        env_id=env_id, save_path=os.path.join(MAIN_DIR, 'Models'))
    #agent.save_models()
    #agent.save_replay_memory()
    n_games = 10
    filename = env_id + '_'+ str(n_games) + '_clamp_on_sigma.png'
    figure_file = os.path.join(MAIN_DIR, 'plots/' + filename)

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True
    if load_checkpoint:
        agent.load_models()
        #env.render(mode='human')
    #agent.load_models()
    steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            if not load_checkpoint:
                action = agent.get_action(observation)
            else:
                action = agent.get_action(observation, deterministic=True)
            observation_, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            else:
                env.render()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-HIST_LEN:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                pass #agent.save_models()

        print(f'Episode {i} | Score {round(score, 1)} | {HIST_LEN} Game Avg {round(avg_score, 1)} | Steps {steps} | {env_id} | Alpha {agent.alpha.numpy()} | Log Alpha {agent.log_alpha.numpy()}')

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file, hist_len=HIST_LEN)
        
    if not load_checkpoint:
        agent.save_models()
        agent.save_replay_memory()