import math
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

gamma = 0.9
# initialize the game board
board = np.zeros((3, 3))


def is_terminal(board):
    if is_winner(board,1) or is_winner(board,-1) or is_full(board):
        return True
    return False


def is_winner(board, player):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player or board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False


def is_full(board):
    if np.count_nonzero(board) == 9:
        return True
    return False


# Define a function to make a move
def make_move(board, player, row, col,action,reward):
    transition_prob= 1.0 / (9-np.count_nonzero(board))
    # if player==1:
    #     action.append((board.copy(),row,col,transition_prob))
    #     reward.append(1)
    board[row][col] = player

def make_move2(board, player, row, col):
    transition_prob= 1.0 / (9-np.count_nonzero(board))
    # if player==1:
    #     action.append((board.copy(),row,col,transition_prob))
    #     reward.append(1)
    tmp=board.copy()
    tmp[row][col] = player
    return tmp

# Define a function to get the available moves
def get_available_moves(board):
    available_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                available_moves.append((i, j))
    return available_moves

def reward_to_go(reward):
    for i in range(len(reward)-2,-1,-1):
        reward[i]=reward[i]+0.9*reward[i+1]
    return reward


def generate_trajectory():
    board = np.zeros((3, 3))
    # set up the players
    player = 1
    opponent = -1
    action = []
    reward = []
    # We let the opponent plays first. Randomly choose a spot for the opponent
    opponent_row, opponent_col = random.choice(get_available_moves(board))
    make_move(board, -1, opponent_row, opponent_col,action,reward)

    # Play the game until it's over
    while not is_terminal(board):
        # Get the available moves
        available_moves = get_available_moves(board)
        # Choose a random move
        player_row, player_col = random.choice(available_moves)
        # Make the move
        make_move(board, 1, player_row, player_col,action,reward)
        # Choose a random move for the opponent
        opponent_row, opponent_col = random.choice(get_available_moves(board))
        # Make the move
        make_move(board, -1, opponent_row, opponent_col,action,reward)

    if is_terminal(board):
        # If the player wins, set the reward to 10
        if is_winner(board, player):
            reward.append(10)
        # If the opponent wins, set the reward to -10
        elif is_winner(board, opponent):
            reward.append(-10)
        # Otherwise, it's a draw and the reward is 0
        else:
            reward.append(0)
    action.append((board.copy(),-1,-1,1))
    #print(action,reward)
    reward_to_go_list=reward_to_go(reward)
    #print(reward_to_go_list)
    for idx in range(len(action)):
        i = action[idx]
        if idx<len(action)-1:
            print('The state is \n',i[0])
            print(f'You choose row {i[1]} col {i[2]} and transition probability is {i[3]}')
            print(f'The reward-to-go is {reward_to_go_list[idx]}')
            print('--------------')
        else:
            print('The terminal state is \n',i[0])
            print(f'The reward-to-go is {reward_to_go_list[idx]}')




    # If the game is over and there's no winner, it's a draw and the reward is 0
    return reward


def hash_state(state):
    str_state=''
    for i in state:
        str_state+=str(i)
    return str_state


def generate_all_states(board,hash_for_all_states):
    if is_terminal(board):
        hash_for_all_states[hash_state(board)]=[0,board.copy()]
        return
    hash_for_all_states[hash_state(board)] = [0,board.copy()]
    available_moves = get_available_moves(board)
    for row, col in available_moves:
        new_board=make_move2(board.copy(),1,row,col)
        if is_winner(new_board,1):
            hash_for_all_states[hash_state(new_board)]=[0,new_board.copy()]
        else:
            available_moves_oppo = get_available_moves(new_board)
            for row,col in available_moves_oppo:
                board_oppo = make_move2(new_board.copy(),-1,row,col)
                generate_all_states(board_oppo,hash_for_all_states)
    return

f1_board=np.array([[-1,0,-1],[1,1,0],[-1,0,0]])
# generate_all_states(f1_board)


value_change = [0]
def value_iteration(state,hash_for_all_states):
    gamma=0.9
    generate_all_states(state,hash_for_all_states)
    while True:
        U = copy.deepcopy(hash_for_all_states)
        delta = 0
        for state_str in U:
            u, each_state=U[state_str]
            optimal_action = 0
            if is_terminal(each_state):
                if is_winner(each_state, 1):
                    hash_for_all_states[state_str] = [10, copy.deepcopy(each_state)]
                elif is_winner(each_state, -1):
                    hash_for_all_states[state_str] = [-10, copy.deepcopy(each_state)]
                else:
                    hash_for_all_states[state_str] = [0, copy.deepcopy(each_state)]

            else:
                available_moves = get_available_moves(each_state)
                reward = [0]*len(available_moves)
                for idx,move in enumerate(available_moves):
                    myrow=move[0]
                    mycol=move[1]
                    new_board = make_move2(copy.deepcopy(each_state), 1, myrow, mycol)
                    reward_for_this_action=0

                    if is_winner(new_board, 1):
                        reward_for_this_action=U[hash_state(new_board)][0]
                        # hash_for_all_states[hash_state(new_board)] = [10, new_board.copy()]
                    else:
                        available_moves_oppo = get_available_moves(new_board)
                        probability_oppo = 1 / len(available_moves_oppo)
                        for row, col in available_moves_oppo:
                            board_oppo = make_move2(copy.deepcopy(new_board), -1, row, col)
                            reward_for_this_action+=probability_oppo * U[hash_state(board_oppo)][0]
                    reward[idx]=reward_for_this_action
                hash_for_all_states[state_str][0]=1+gamma*max(reward)
            if abs(U[state_str][0] - hash_for_all_states[state_str][0])>delta:
                delta=abs(U[state_str][0] - hash_for_all_states[state_str][0])
                # print('delta change', delta)

            if state_str == hash_state(state):
                value_change.append(hash_for_all_states[state_str][0])
                idx_action=reward.index(max(reward))
                opt_action=available_moves[idx_action]

        if delta<0.1:
            print(opt_action)
            print(value_change)
            plt.plot(range(len(value_change)), value_change)
            plt.title('value iteration for the MDP')
            plt.xlabel('the number of iterations')
            plt.ylabel('Value')
            plt.show()
            break
    return

hash_for_all_states={}
# value_iteration(f1_board,hash_for_all_states)
f2_board=np.array([[1,-1,1],[0,0,1],[-1,1,-1]])
f3_board=np.array([[-1,-1,1],[1,1,-1],[-1,0,0]])
value_iteration(f2_board,hash_for_all_states)
# print(hash_for_all_states)

# print(hash_for_all_states[hash_state(f1_board)])