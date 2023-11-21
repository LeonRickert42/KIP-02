import numpy as np
import gymnasium as gym
import pickle

def policy_evaluation(env, policy, gamma=0.9, epsilon=1e-6):
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            a = policy[s]
            bellman_backup = sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a])
            V[s] = bellman_backup
            delta = max(delta, abs(v - V[s]))

        if delta < epsilon:
            break

    return V

def policy_improvement(env, V, gamma=0.9):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        q_values = [sum(p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]) for a in range(num_actions)]
        policy[s] = np.argmax(q_values)

    return policy

def policy_iteration(env, gamma=0.9, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.zeros(num_states, dtype=int)

    while True:
        V = policy_evaluation(env, policy, gamma, epsilon)
        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy

def main():
    #env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    env = gym.make('Taxi-v3', render_mode='human')
    optimal_values, optimal_policy = policy_iteration(env)
    terminated = False
    truncated = False
    state = env.reset()[0]
    while (not terminated and not truncated):
        action = optimal_policy[state]
        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
    env.close()

    print("Optimal Values:")
    print(optimal_values)

    print("\nOptimal Policy:")
    print(optimal_policy)

if __name__ == "__main__":
    main()
