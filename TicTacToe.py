# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 2025

@author: Refactored_AI
Theme: Cyberpunk / Hacker Interface
"""

import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox
import time

# ==========================================
# 1. Game Constants & Configuration
# ==========================================
BOARD_ROWS = 3
BOARD_COLS = 3
PLAYER_X = 1   # Human (NetRunner)
PLAYER_O = -1  # AI (Core System)
EMPTY = 0

# Cyberpunk Styling
COLOR_BG = "#0d1117"       # Dark background
COLOR_BTN = "#161b22"      # Button background
COLOR_HUMAN = "#00ff41"    # Neon Green
COLOR_AI = "#ff003c"       # Neon Red
FONT_MAIN = ("Consolas", 20, "bold")
FONT_TITLE = ("Courier New", 16, "bold")

# ==========================================
# 2. Reinforcement Learning Agent
# ==========================================
class QLearningAgent:
    def __init__(self, name, epsilon=0.3):
        self.name = name
        self.states = []  # History of positions in current game
        self.lr = 0.2     # Learning Rate
        self.decay_gamma = 0.9
        self.epsilon = epsilon # Exploration rate
        self.states_value = {} # Q-Table (The Brain)

    def get_hash(self, board):
        """Converts board matrix to a string hash for dictionary lookup."""
        return str(board.reshape(BOARD_COLS * BOARD_ROWS))

    def choose_action(self, available_positions, current_board, symbol):
        """Decides move based on Exploration (Random) or Exploitation (Q-Table)."""
        
        # Exploration: Try random move (only during training)
        if np.random.uniform(0, 1) <= self.epsilon:
            idx = np.random.choice(len(available_positions))
            action = available_positions[idx]
        else:
            # Exploitation: Choose best move from Q-Table
            value_max = -999
            action = None
            
            # If no positions available (should be handled before), return None
            if not available_positions:
                return None
                
            # Default action if dictionary is empty
            action = available_positions[0] 

            for p in available_positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_board_hash = self.get_hash(next_board)
                
                value = 0 if self.states_value.get(next_board_hash) is None else self.states_value.get(next_board_hash)
                
                if value >= value_max:
                    value_max = value
                    action = p
        
        return action

    def add_state(self, state_hash):
        self.states.append(state_hash)

    def feed_reward(self, reward):
        """Backpropagates the reward to update Q-values."""
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            
            # Q-Learning Formula
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset_history(self):
        self.states = []

    def save_policy(self):
        with open('policy_' + str(self.name), 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def load_policy(self, file_name):
        try:
            with open(file_name, 'rb') as fr:
                self.states_value = pickle.load(fr)
        except FileNotFoundError:
            print("No saved policy found. Agent will play randomly/untrained.")

# ==========================================
# 3. Game Environment (Logic)
# ==========================================
class TicTacToeEnvironment:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.is_end = False
        self.winner_symbol = None

    def get_hash(self):
        return str(self.board.reshape(BOARD_COLS * BOARD_ROWS))

    def available_positions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def update_state(self, position, player_symbol):
        self.board[position] = player_symbol

    def check_winner(self):
        # Check Rows & Cols
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3 or sum(self.board[:, i]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[i, :]) == -3 or sum(self.board[:, i]) == -3:
                self.is_end = True
                return -1
        
        # Check Diagonals
        diag1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        
        if diag1 == 3 or diag2 == 3:
            self.is_end = True
            return 1
        if diag1 == -3 or diag2 == -3:
            self.is_end = True
            return -1

        # Check Draw
        if len(self.available_positions()) == 0:
            self.is_end = True
            return 0 # Draw
            
        self.is_end = False
        return None

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.is_end = False
        self.winner_symbol = None

# ==========================================
# 4. GUI Interface (Cyberpunk Theme)
# ==========================================
class CyberpunkGameGUI:
    def __init__(self, master, ai_agent):
        self.master = master
        self.master.title("CYBERPUNK // TIC-TAC-TOE")
        self.master.configure(bg=COLOR_BG)
        
        self.ai_agent = ai_agent
        self.env = TicTacToeEnvironment()
        self.human_symbol = 1   # NetRunner
        self.ai_symbol = -1     # Core System
        
        # UI Setup
        self.create_widgets()
        self.reset_game()

    def create_widgets(self):
        # Header
        self.lbl_status = tk.Label(self.master, text="NETRUNNER VS CORE_SYSTEM", 
                                   font=FONT_TITLE, bg=COLOR_BG, fg=COLOR_HUMAN)
        self.lbl_status.pack(pady=10)

        # Game Grid Frame
        self.frame_grid = tk.Frame(self.master, bg=COLOR_BG)
        self.frame_grid.pack()

        # Buttons (3x3)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                btn = tk.Button(self.frame_grid, text="", font=FONT_MAIN, width=5, height=2,
                                bg=COLOR_BTN, fg="white", activebackground="#30363d",
                                borderwidth=0, command=lambda row=r, col=c: self.on_human_move(row, col))
                btn.grid(row=r, column=c, padx=5, pady=5)
                self.buttons[r][c] = btn

        # Footer Actions
        self.btn_reset = tk.Button(self.master, text="REBOOT SYSTEM", font=("Consolas", 12),
                                   bg="#21262d", fg="white", command=self.reset_game)
        self.btn_reset.pack(pady=15)

    def on_human_move(self, row, col):
        """Triggered when Human clicks a button."""
        if self.env.is_end or self.env.board[row, col] != 0:
            return

        # 1. Update Internal State
        self.env.update_state((row, col), self.human_symbol)
        
        # 2. Update Visuals
        self.buttons[row][col].config(text="X", fg=COLOR_HUMAN, disabledforeground=COLOR_HUMAN)
        
        # 3. Check Status
        result = self.env.check_winner()
        if result is not None:
            self.game_over(result)
        else:
            # 4. Trigger AI Turn
            self.lbl_status.config(text="CORE_SYSTEM CALCULATING...", fg=COLOR_AI)
            self.master.after(500, self.ai_move) # Small delay for realism

    def ai_move(self):
        """Logic for AI to take its turn."""
        available = self.env.available_positions()
        
        # AI chooses action based on policy
        action = self.ai_agent.choose_action(available, self.env.board, self.ai_symbol)
        
        if action:
            # 1. Update State
            self.env.update_state(action, self.ai_symbol)
            
            # 2. Update Visuals
            r, c = action
            self.buttons[r][c].config(text="O", fg=COLOR_AI, disabledforeground=COLOR_AI)
            
            # 3. Check Status
            result = self.env.check_winner()
            if result is not None:
                self.game_over(result)
            else:
                self.lbl_status.config(text="YOUR TURN, NETRUNNER", fg=COLOR_HUMAN)

    def game_over(self, result):
        if result == 1:
            msg = "SYSTEM BREACHED. YOU WIN."
            color = COLOR_HUMAN
        elif result == -1:
            msg = "ACCESS DENIED. AI WINS."
            color = COLOR_AI
        else:
            msg = "DEADLOCK DETECTED. DRAW."
            color = "white"
        
        self.lbl_status.config(text=msg, fg=color)
        messagebox.showinfo("Game Over", msg)

    def reset_game(self):
        self.env.reset()
        self.lbl_status.config(text="NETRUNNER INITIATED", fg=COLOR_HUMAN)
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="", bg=COLOR_BTN)

# ==========================================
# 5. Training Phase (Before GUI opens)
# ==========================================
def train_agent(rounds=10000):
    print(f"Training Agent for {rounds} rounds... Please wait.")
    
    p1 = QLearningAgent("p1", epsilon=0.3)
    p2 = QLearningAgent("p2", epsilon=0.3)
    env = TicTacToeEnvironment()
    
    for i in range(rounds):
        # Progress indicator
        if i % 2000 == 0:
            print(f"Training progress: {(i/rounds)*100:.0f}%")
            
        current_symbol = 1 # P1 starts
        
        while not env.is_end:
            # Determine current player
            if current_symbol == 1:
                agent = p1
            else:
                agent = p2
                
            avail_pos = env.available_positions()
            action = agent.choose_action(avail_pos, env.board, current_symbol)
            
            # Update state
            env.update_state(action, current_symbol)
            agent.add_state(env.get_hash())
            
            win = env.check_winner()
            
            if win is not None:
                # Distribute Rewards
                if win == 1:
                    p1.feed_reward(1)
                    p2.feed_reward(0)
                elif win == -1:
                    p1.feed_reward(0)
                    p2.feed_reward(1)
                else: # Tie
                    p1.feed_reward(0.1)
                    p2.feed_reward(0.5)
                
                # Reset logic
                p1.reset_history()
                p2.reset_history()
                env.reset()
                break
                
            # Switch turn
            current_symbol = -1 if current_symbol == 1 else 1

    print("Training Complete!")
    return p2 # Return p2 (the AI playing as O)

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Train the AI briefly so it knows how to play
    # Note: We train 10k rounds so it starts quickly. For perfect play, use 50k+.
    trained_ai = train_agent(10000)
    
    # 2. Switch AI to exploitation mode (no random moves)
    trained_ai.epsilon = 0 
    
    # 3. Launch GUI
    root = tk.Tk()
    # Center the window roughly
    root.geometry("400x450")
    app = CyberpunkGameGUI(root, trained_ai)
    root.mainloop()