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


def value_iteration(S, A, P, R, gamma=0.9):
    """
    :param list S: set of states
    :param list A: set of actions
    :param function P: transition function
    :param function R: reward function
    :param float gamma: discount factor
    :return: dictionary representing the optimal value function V
    """

    # Initialize the value function V with zeros for all states
    V = {s: 0 for s in S}

    # Continue iterating until the value function converges
    while True:
        # Make a copy of the current value function for comparison later
        oldV = V.copy()

        # Update the value function for each state
        for s in S:
            # Skip updating terminal states
            if env.unwrapped.P[s][0][0][0] is None:
                continue

            # Calculate the Q-values for each action in the current state
            Q = {a: R(s, a) + gamma * sum(P(s_next, s, a) * oldV[s_next] for s_next in S) for a in A}

            # Update the value function of the current state to be the maximum Q-value
            V[s] = max(Q.values())

        # Check for convergence by comparing the new and old value functions
        if all(oldV[s] == V[s] for s in S):
            break

    # Return the optimal value function V
    return V


# Create FrozenLake environment
env = gym.make('FrozenLake-v1', render_mode='rgb_array')

# Extract parameters from the environment
S, A, P, R = extract_parameters_from_environment(env)

# Perform Value Iteration
optimal_value_function = value_iteration(S, A, P, R)

print("Optimal Value Function:")
print(optimal_value_function)

# Visualize the Frozen Lake with the optimal policy
state = env.reset()
done = False

while not done:
    # Render the current state
    env.render()

    # Take the action with the maximum Q-value in the current state
    action = max(A, key=lambda a: R(state, a) + 0.9 * sum(
        P(s_next, state, a) * optimal_value_function[s_next] for s_next in S))

    # Take the action and observe the next state and reward
    state, reward, done, _, _ = env.step(action)

# Close the environment rendering
env.close()
