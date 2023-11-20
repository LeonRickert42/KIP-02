import gymnasium as gym
import pygame


def extract_parameters_from_environment(env):
    """
    Extract necessary parameters from the environment.
    :param env: OpenAI Gym environment
    :return: S, A, P, R
    """
    # Set of states
    S = list(range(env.observation_space.n))

    # Set of actions
    A = list(range(env.action_space.n))

    # Transition function
    def P(s_next, s, a):
        transitions = env.unwrapped.P[s][a]
        return sum(prob * (next_s == s_next) for prob, next_s, _, _ in transitions)

    # Reward function
    def R(s, a):
        return env.unwrapped.P[s][a][0][2]

    return S, A, P, R


def policy_evaluation(policy, S):
    V = {s: 0 for s in S}
    while True:
        oldV = V.copy()

        for s in S:
            a = policy[s]
            V[s] = R(s, a) + sum(P(s_next, s, a) * oldV[s_next]
                                 for s_next in S)

        if all(oldV[s] == V[s] for s in S):
            break
    return V


def policy_improvement(V, S, A):
    policy = {s: A[0] for s in S}

    for s in S:
        Q = {}
        for a in A:
            Q[a] = R(s, a) + sum(P(s_next, s, a) * V[s_next]
                                 for s_next in S)

        policy[s] = max(Q, key=Q.get)

    return policy


def policy_iteration(S, A, P, R):
    """
        :param list S: set of states
        :param list A: set of actions
        :param function P: transition function
        :param function R: reward function
    """
    policy = {s: A[0] for s in S}

    while True:
        old_policy = policy.copy()
        V = policy_evaluation(policy, S)
        policy = policy_improvement(V, S, A)
        if all(old_policy[s] == policy[s] for s in S):
            break
    return policy


# Create FrozenLake environment
env = gym.make('FrozenLake-v1', render_mode='rgb_array')

# Extract parameters from the environment
S, A, P, R = extract_parameters_from_environment(env)

# Perform Policy Iteration
optimal_policy = policy_iteration(S, A, P, R)

print("Optimal Policy:")
print(optimal_policy)
