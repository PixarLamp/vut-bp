# author: Zuzana Hrkľová
# year: 2023
# file: main.py
# description: Implementation of functions performing one move of agents/Mr.X using a method of choice 

# Using of Michal Sova's mc_environment edited by Martin Gerža and Martin Gerža's version of MCTS
from MC_2022 import mc_environment, MG_monte_carlo

# Using of Michal Sova's (2021) version of alfa_beta_wrapper and player
from MS_2021 import alfa_beta_wrapper, player, monte_carlo

import DQN_agent
import DQN_mrx

import torch
import numpy as np

from datetime import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"

env = mc_environment.Environment()
level = -1

# performns one move of Mr.X using DQN method
def dqn_mrx_move(mrx_pos, agents, field_size, model):
    
    state = DQN_mrx.get_state(field_size, mrx_pos, agents)
    state1 = torch.tensor(state)
    state1 = state1.to(torch.device(device))  
    qval = model(state1.float()).data.cpu().numpy()
    action = np.argmax(qval)
    chosen_action = DQN_agent.actions[action]
    
    move_eval, next_move = DQN_mrx.move_mrx(chosen_action, field_size, mrx_pos, agents)
    return next_move

# performs one move with all the agents using DQN method
def move_all_DQN_agents(mrx_probability, agents, field_size, model):
    current_agents = []
    for agent in env.agents:
        current_agents.append(agent)
    for agent in current_agents:
        agents.remove(agent)
        grid = DQN_agent.create_grid(mrx_probability, agent, field_size, agents)
        grid_size = field_size*field_size*2 
        state = grid.reshape(1, grid_size)
        state1 = torch.tensor(state)
        state1 = state1.to(torch.device(device))    
        qval = model(state1.float()).data.cpu().numpy()
        action = np.argmax(qval)
        chosen_action = DQN_agent.actions[action]
        move_stat, agents = DQN_agent.move_NN_agent(agent, chosen_action, field_size, agents)
    return agents

#returns one Mr.X move performed using method of choice 
def move_mrx(move_num, mrx_type, mrx_pos, agents, field_size, moves, last_seen_length, sim_time, model):
    mrx_move = []
    env.field_size = field_size
    env.agents = agents
    if mrx_type == 0:
        # DQN method by Zuzana Hrklova 2023
        mrx_move = dqn_mrx_move(mrx_pos, agents, field_size, model)
    
    elif mrx_type == 1:
        # Alpha-Beta method by Andrej Tulusak 2020
        mrx_move = alfa_beta_wrapper.mrx_move(agents, mrx_pos, field_size)
    
    elif mrx_type == 2:
        # Monte Carlo Tree Search method by Martin Gerza 2022
        mrx_move = MG_monte_carlo.mc_mrX_move(mrx_pos, last_seen_length, agents, move_num, sim_time, level, moves, field_size)
    
    elif mrx_type == 3:
        # Monte Carlo Tree Search method by Michal Sova 2021
        mrx_move = monte_carlo.mc_mrx_move(mrx_pos, agents, move_num, sim_time, level, moves, field_size)
    
    else:
        # random Mr.X movement by Michal Sova 
        mrx_move = monte_carlo.random_mrx_moves(mrx_pos, field_size)

    if not env.move(0, mrx_move):
        print('Move ' + str(mrx_move) + ' of Mr. X not possible!')
        return False

    return True, mrx_move

#returns one move of all agents performed using method of choice 
def move_agents(move_num, agents_type, agents, field_size, mrx_probability, moves, sim_time, mrx_last_seen, last_seen_length, mrx_ls_pos, mrx_pos, model):
    next_moves = []
    
    if agents_type == 0:
        # DQN method by Zuzana Hrklova 2023
        next_moves = move_all_DQN_agents(mrx_probability, agents, field_size, model)

    elif agents_type == 1:
        # Alpha-Beta method by Andrej Tulusak 2020
        if alfa_beta_wrapper.a_b is not None and alfa_beta_wrapper.a_b.root is not None and move_num == 0:
            alfa_beta_wrapper.a_b.root.reset()
        next_moves = alfa_beta_wrapper.agents_move(agents, mrx_pos, field_size, mrx_last_seen % last_seen_length)
    
    elif agents_type == 2:
        # Monte Carlo Tree Search method by Martin Gerza 2022
        next_moves = MG_monte_carlo.mc_agents_move(mrx_ls_pos, last_seen_length, agents, move_num, sim_time, level, moves, field_size)
    
    elif agents_type == 3:
        # Monte Carlo Tree Search method by Michal Sova 2021
        next_moves = monte_carlo.mc_agents_move(mrx_last_seen, mrx_ls_pos, last_seen_length, agents, move_num, sim_time, level, moves, field_size)
    
    else:
        # random agent movement by Michal Sova 
        next_moves = monte_carlo.random_agent_moves(agents, field_size)

    return True, next_moves