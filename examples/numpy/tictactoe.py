import streamlit as st
import numpy as np
import time

# ==========================================
# 1. DP ENGINE & ENVIRONMENT (Backend)
# ==========================================

class TabularTicTacToe:
    def __init__(self, opponent_policy='random'):
        self.opponent_policy = opponent_policy
        self.p1 = 1   # Agent (X)
        self.p2 = -1  # Opponent (O)
        self.state_to_id = {}
        self.id_to_state = {}
        self.P = {} 
        self._generate_state_space()

    def _get_valid_actions(self, board):
        return [i for i, x in enumerate(board) if x == 0]

    def _check_winner(self, board):
        b = np.array(board).reshape(3, 3)
        lines = ([b[i, :] for i in range(3)] + [b[:, i] for i in range(3)] + 
                 [b.diagonal(), np.fliplr(b).diagonal()])
        for line in lines:
            if np.all(line == self.p1): return self.p1
            if np.all(line == self.p2): return self.p2
        if 0 not in board: return 0 
        return None 

    def _opponent_move(self, board):
        # Determine opponent behavior for TRAINING
        actions = self._get_valid_actions(board)
        if not actions: return []
        
        # If Random, uniform probability over all empty spots
        if self.opponent_policy == 'random':
            prob = 1.0 / len(actions)
            return [(prob, tuple([self.p2 if i == a else x for i, x in enumerate(board)])) for a in actions]
        
        # If Optimal (Minimax-lite), block wins or pick random
        # (Simplified to ensure the agent learns to defend)
        # For this demo, we treat 'optimal' as a stricter version of random that never misses a win.
        best_outcomes = []
        # Check if opponent can win immediately
        for action in actions:
            temp_board = list(board)
            temp_board[action] = self.p2
            if self._check_winner(temp_board) == self.p2:
                return [(1.0, tuple(temp_board))] # Opponent takes the win 100%

        # Otherwise random
        prob = 1.0 / len(actions)
        return [(prob, tuple([self.p2 if i == a else x for i, x in enumerate(board)])) for a in actions]

    def _generate_state_space(self):
        start_board = tuple([0] * 9)
        self.state_to_id[start_board] = 0
        self.id_to_state[0] = start_board
        queue = [start_board]
        curr_id = 0
        
        while queue:
            s_board = queue.pop(0)
            s_id = self.state_to_id[s_board]
            self.P[s_id] = {a: [] for a in range(9)}
            
            if self._check_winner(s_board) is not None: continue

            for action in self._get_valid_actions(s_board):
                # 1. Agent Move
                next_b = list(s_board)
                next_b[action] = self.p1
                next_b_tuple = tuple(next_b)
                
                winner = self._check_winner(next_b_tuple)
                if winner == self.p1:
                    self.P[s_id][action].append((1.0, -1, 1, True)) # Win
                    continue
                elif winner == 0:
                    self.P[s_id][action].append((1.0, -1, 0, True)) # Draw
                    continue
                
                # 2. Opponent Response (Environment)
                outcomes = self._opponent_move(next_b_tuple)
                for prob, final_board in outcomes:
                    opp_w = self._check_winner(final_board)
                    if opp_w == self.p2:
                        self.P[s_id][action].append((prob, -1, -1, True)) # Loss
                    elif opp_w == 0:
                        self.P[s_id][action].append((prob, -1, 0, True)) # Draw
                    else:
                        if final_board not in self.state_to_id:
                            curr_id += 1
                            self.state_to_id[final_board] = curr_id
                            self.id_to_state[curr_id] = final_board
                            queue.append(final_board)
                        self.P[s_id][action].append((prob, self.state_to_id[final_board], 0, False))

@st.cache_resource
def train_agent(opponent_type):
    """
    Runs Value Iteration once and caches the result.
    """
    env = TabularTicTacToe(opponent_policy=opponent_type)
    V = np.zeros(len(env.state_to_id))
    gamma = 0.9
    theta = 1e-4
    
    # Value Iteration Loop
    iteration = 0
    while True:
        delta = 0
        for s in range(len(env.state_to_id)):
            v = V[s]
            actions = env.P[s].keys()
            # Max over actions
            max_val = -float('inf')
            has_actions = False
            for a in actions:
                transitions = env.P[s][a]
                if not transitions: continue
                has_actions = True
                expected_val = 0
                for prob, next_s, r, done in transitions:
                    val_next = 0 if done else V[next_s]
                    expected_val += prob * (r + gamma * val_next)
                if expected_val > max_val:
                    max_val = expected_val
            
            V[s] = max_val if has_actions else 0
            delta = max(delta, abs(v - V[s]))
        
        iteration += 1
        if delta < theta: break
        
    return env, V

# ==========================================
# 2. FRONTEND LOGIC (Streamlit)
# ==========================================

st.set_page_config(page_title="DP Tic-Tac-Toe", layout="centered")

st.title("🤖 Tic-Tac-Toe via Dynamic Programming")
st.markdown("""
This is not a standard hard-coded AI. This agent **learned** the game moments ago by solving the 
Bellman Optimality Equation: $V(s) = \max_a \sum P(s'|s,a)[R + \gamma V(s')]$.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    opp_mode = st.selectbox("Training Opponent Assumption", ["random", "optimal"], 
                            help="If 'random', AI assumes you might make mistakes. If 'optimal', AI plays defensively assuming you are perfect.")
    
    if st.button("Reset / Retrain"):
        st.cache_resource.clear()
        if 'board' in st.session_state:
            del st.session_state['board']
        st.rerun()

# --- Initialize Game State ---
if 'board' not in st.session_state:
    st.session_state.board = [0] * 9
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.turn = 'Agent' # Agent moves first

# --- Load/Train Agent ---
with st.spinner(f"Running Value Iteration (Training against '{opp_mode}' opponent)..."):
    env, V = train_agent(opp_mode)

# --- Helper Functions ---
def get_ai_move(board, env, V):
    """
    Uses the computed Value Function V to select the best move.
    """
    s_tuple = tuple(board)
    if s_tuple not in env.state_to_id:
        # Should not happen if Agent starts, but fallback to random if state unseen
        return np.random.choice([i for i, x in enumerate(board) if x == 0])
    
    s_id = env.state_to_id[s_tuple]
    best_action = -1
    best_value = -float('inf')
    
    # Evaluate valid actions based on V
    for action in env.P[s_id]:
        transitions = env.P[s_id][action]
        if not transitions: continue
        
        # Q-value calculation: Sum(Prob * (Reward + Gamma * V_next))
        q_val = 0
        for prob, next_s, r, done in transitions:
            val_next = 0 if done else V[next_s]
            q_val += prob * (r + 0.9 * val_next)
            
        if q_val > best_value:
            best_value = q_val
            best_action = action
            
    return best_action

def check_game_status(board):
    # 1 = Agent (X), -1 = Human (O)
    b = np.array(board).reshape(3, 3)
    # Check rows, cols, diags
    lines = ([b[i, :] for i in range(3)] + [b[:, i] for i in range(3)] + 
             [b.diagonal(), np.fliplr(b).diagonal()])
    
    for line in lines:
        if np.all(line == 1): return "Agent Wins!"
        if np.all(line == -1): return "You Win!"
    if 0 not in board: return "Draw"
    return None

# --- Game Loop Logic ---

# 1. If it's Agent's turn and game not over
if st.session_state.turn == 'Agent' and not st.session_state.game_over:
    move = get_ai_move(st.session_state.board, env, V)
    st.session_state.board[move] = 1 # Agent is 1
    
    res = check_game_status(st.session_state.board)
    if res:
        st.session_state.game_over = True
        st.session_state.winner = res
    else:
        st.session_state.turn = 'Human'
    st.rerun()

# 2. Render Board
st.write(f"**Status:** {st.session_state.winner if st.session_state.game_over else f'{st.session_state.turn} Turn'}")

# Grid layout for board
# We use columns to create the 3x3 grid
for i in range(0, 9, 3):
    cols = st.columns(3)
    for j in range(3):
        idx = i + j
        val = st.session_state.board[idx]
        
        # Determine label and style
        if val == 1: label = "❌"
        elif val == -1: label = "⭕"
        else: label = " "
        
        # Interactive Button logic
        # Button is disabled if spot is taken OR if game is over OR if it's not Human turn
        is_disabled = (val != 0) or st.session_state.game_over or (st.session_state.turn != 'Human')
        
        if cols[j].button(label, key=f"btn_{idx}", disabled=is_disabled, use_container_width=True):
            # Human Move
            st.session_state.board[idx] = -1
            res = check_game_status(st.session_state.board)
            if res:
                st.session_state.game_over = True
                st.session_state.winner = res
            else:
                st.session_state.turn = 'Agent'
            st.rerun()

# --- Debug/Explanation ---
with st.expander("Peek into the Agent's Brain"):
    curr_tuple = tuple(st.session_state.board)
    if curr_tuple in env.state_to_id and not st.session_state.game_over:
        sid = env.state_to_id[curr_tuple]
        st.write(f"Current State ID: `{sid}`")
        st.write(f"Computed Value V(s): `{V[sid]:.4f}`")
        st.write("Expected Returns for possible moves:")
        
        # Show Q-values for next moves
        action_data = {}
        for a in env.P[sid]:
            if not env.P[sid][a]: continue
            q = sum([p * (r + 0.9 * (0 if d else V[ns])) for p, ns, r, d in env.P[sid][a]])
            action_data[f"Cell {a}"] = q
        st.bar_chart(action_data)
    else:
        st.write("Game Over or Unknown State")