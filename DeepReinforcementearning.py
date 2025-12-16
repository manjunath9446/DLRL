# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning: Q-Learning on a Graph
Theme: Rescue Robot Navigation
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. Graph Setup & Visualization
# ==========================================

# Define connections between rooms (nodes)
# (Start_Node, End_Node)
edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), 
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9)]

goal_node = 10
node_count = 11

G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42) # Seed for consistent layout

print("Displaying Map...")
plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.title("Building Map Layout")
plt.show()

# ==========================================
# 2. Reward Matrix Initialization
# ==========================================

# Initialize Reward Matrix 'R' with -1 (Impossible moves)
# We use np.array instead of np.matrix as it is the modern standard
R = np.ones(shape=(node_count, node_count)) * -1

# Update R matrix based on edges
for point in edges:
    # Forward connection
    if point[1] == goal_node:
        R[point] = 100
    else:
        R[point] = 0

    # Backward connection (undirected graph)
    if point[0] == goal_node:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0

# The goal node loops to itself with high reward
R[goal_node, goal_node] = 100

# ==========================================
# 3. Q-Learning (Basic Phase)
# ==========================================

Q = np.zeros([node_count, node_count])
gamma = 0.8 # Discount factor (importance of future rewards)
alpha = 1.0 # Learning rate

def get_available_actions(state):
    """Returns list of connected nodes for a given state."""
    current_state_row = R[state, :]
    # Find indices where value is >= 0 (valid moves)
    av_act = np.where(current_state_row >= 0)[0]
    return av_act

def sample_next_action(available_actions_list):
    """Randomly chooses one action from available options."""
    next_action = np.random.choice(available_actions_list)
    return next_action

def update_q(current_state, action, gamma):
    """Updates the Q-table using the Bellman Equation."""
    # Max value of the *next* state's possible actions
    max_index = np.where(Q[action, :] == np.max(Q[action, :]))[0]
    
    # If multiple actions have max value, pick random
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index))
    else:
        max_index = int(max_index)
        
    max_value = Q[action, max_index]
    
    # Q-Learning Formula
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    
    # Return normalized score for plotting
    if np.max(Q) > 0:
        return (np.sum(Q / np.max(Q) * 100))
    else:
        return 0

# --- Training Loop 1 (Basic Pathfinding) ---
print("Training Phase 1: Basic Navigation...")
scores = []
for i in range(700):
    # Pick random start state
    current_state = np.random.randint(0, node_count)
    available_act = get_available_actions(current_state)
    action = sample_next_action(available_act)
    score = update_q(current_state, action, gamma)
    scores.append(score)

print("Training Complete.")

# --- Testing Phase 1 ---
current_state = 0
steps = [current_state]

while current_state != goal_node:
    # Choose best action from Q table
    next_step_index = np.where(Q[current_state, :] == np.max(Q[current_state, :]))[0]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index
    
    # Break infinite loop if training failed
    if len(steps) > 20:
        break

print(f"Shortest path found: {steps}")

plt.plot(scores)
plt.xlabel('Iterations')
plt.ylabel('Total Reward')
plt.title('Training Progress (Phase 1)')
plt.show()

# ==========================================
# 4. Environment Awareness (Hazards & Bonuses)
# ==========================================

# Define special nodes
# 'Fire' nodes (Dangerous) - Previously "Police"
hazards_fire = [2, 4, 5] 

# 'Water' nodes (Beneficial) - Previously "Drug Traces"
bonus_water = [3, 8, 9]

print("\n--- Introducing Environmental Factors ---")
print(f"Avoiding Fire/Debris at nodes: {hazards_fire}")
print(f"Collecting Water at nodes: {bonus_water}")

# Mapping for visualization
mapping = {
    0: '0-Start', 
    1: '1', 
    2: '2-Fire', 
    3: '3-Water',
    4: '4-Fire', 
    5: '5-Fire', 
    6: '6', 
    7: '7', 
    8: 'Water',
    9: '9-Water', 
    10: '10-Survivor'
}
H = nx.relabel_nodes(G, mapping)
pos_h = nx.spring_layout(H, seed=42)

plt.figure(figsize=(10,6))
# Draw nodes with different colors based on type
node_colors = []
for node in H.nodes():
    if "Fire" in str(node): node_colors.append('red')
    elif "Water" in str(node): node_colors.append('blue')
    elif "Survivor" in str(node): node_colors.append('green')
    elif "Start" in str(node): node_colors.append('yellow')
    else: node_colors.append('lightgrey')

nx.draw_networkx_nodes(H, pos_h, node_color=node_colors, node_size=600)
nx.draw_networkx_edges(H, pos_h)
nx.draw_networkx_labels(H, pos_h)
plt.title("Environment Map: Red=Avoid, Blue=Collect, Green=Goal")
plt.show()

# 5. Advanced Q-Learning (Environment Aware)

# Reset Q for fresh learning
Q = np.zeros([node_count, node_count])

# Matrices to track what the agent encounters
env_fire_log = np.zeros([node_count, node_count])
env_water_log = np.zeros([node_count, node_count])

def collect_env_data(action_node):
    """Simulate sensors detecting fire or water."""
    data = []
    if action_node in hazards_fire:
        data.append('fire')
    if action_node in bonus_water:
        data.append('water')
    return data

def update_q_advanced(current_state, action, gamma):
    max_index = np.where(Q[action, :] == np.max(Q[action, :]))[0]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index))
    else:
        max_index = int(max_index)
        
    max_value = Q[action, max_index]
    
    # Calculate base reward
    reward = R[current_state, action]
    
    # Get environment feedback
    env_data = collect_env_data(action)
    
    # Apply Environmental Penalties/Bonuses directly to Reward calculation
    # (Note: In the original code, this was just logging. Here we actually influence learning)
    if 'fire' in env_data:
        reward -= 50  # Huge penalty for fire
        env_fire_log[current_state, action] += 1
    if 'water' in env_data:
        reward += 20  # Bonus for water
        env_water_log[current_state, action] += 1
        
    Q[current_state, action] = reward + gamma * max_value
    
    if np.max(Q) > 0:
        return (np.sum(Q / np.max(Q) * 100))
    else:
        return 0

# --- Training Loop 2 (Advanced) ---
print("Training Phase 2: Advanced Navigation...")
scores = []
for i in range(1000):
    current_state = np.random.randint(0, node_count)
    available_act = get_available_actions(current_state)
    action = sample_next_action(available_act)
    score = update_q_advanced(current_state, action, gamma)
    scores.append(score)

print("\nFire Encounters logged:\n", env_fire_log)
print("\nWater Encounters logged:\n", env_water_log)

# --- Final Path Calculation ---
current_state = 0
final_path = [current_state]

while current_state != goal_node:
    # Choose best action
    next_step_index = np.where(Q[current_state, :] == np.max(Q[current_state, :]))[0]
    
    # If multiple best paths, pick one
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index))
    else:
        next_step_index = int(next_step_index)
    
    final_path.append(next_step_index)
    current_state = next_step_index
    
    if len(final_path) > 20: break # Safety break

print("\nMost Efficient Safety Path (Avoiding Fire, finding Water):")
print(final_path)

# Visualize Final Learning Curve
plt.plot(scores)
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Training Progress (Phase 2 - With Env Factors)')
plt.show()

# Visualize Final Path on Graph
path_edges = list(zip(final_path, final_path[1:]))
plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G, pos, node_color='lightgrey', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='grey')
nx.draw_networkx_labels(G, pos)
# Highlight path
nx.draw_networkx_nodes(G, pos, nodelist=final_path, node_color='cyan', node_size=600)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
plt.title("Final Optimal Path (Red)")
plt.show()