# author: Zuzana Hrkľová
# year: 2023
# file: main.py
# description: Implementation of main file for running program

# Using of Michal Sova's mc_environment edited by Martin Gerža and Martin Gerža's version of MCTS
from MC_2022 import mc_environment, MG_monte_carlo

# Using of Michal Sova's (2021) version of alfa_beta_wrapper and player
from MS_2021 import alfa_beta_wrapper, player, monte_carlo

import DQN_agent
import DQN_mrx
import player_method

import argparse
import argparser
import torch
from datetime import datetime

env = mc_environment.Environment()

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
args = None

# device used by neural networks
device = "cuda" if torch.cuda.is_available() else "cpu"

# plays one full game of Scotland Yard
def play_game(mrx_model, agent_model):
    mrx_probability = []
    
    for move in range(args.moves):
        valid_move, new_position = player_method.move_mrx(move, args.mrx_type, env.mrx_pos, env.agents, args.field_size, args.moves, env.last_seen_length, args.sim_time, mrx_model)
        if not valid_move:
            if args.print:
                print('move: ' + str(move + 1))
                env.print()
            return True
        env.mrx_pos = new_position
            
        if ((move + 1) % 3 == 0):
            env.mrx_visible = True
            env.mrx_last_seen = 1
            env.mrx_ls_pos = env.mrx_pos
        else:
            env.mrx_visible = False
            env.mrx_last_seen += 1
                
        if env.win_condition():
            if args.print:
                print('move: ' + str(move + 1))
                env.print()
            return True
            
        mrx_probability = DQN_agent.mrx_position_probability(mrx_probability, env.mrx_visible, env.mrx_pos, env.agents, env.field_size)    
        
        move_made, agents_new_pos = player_method.move_agents(move, args.agents_type, env.agents, env.field_size, mrx_probability, args.moves, args.sim_time, env.mrx_last_seen, env.last_seen_length, env.mrx_ls_pos, env.mrx_pos, agent_model)
        if not move_made:
            if args.print:
                print('move: ' + str(move + 1))
                env.print()
            return False
        env.agents = agents_new_pos
            
        if env.win_condition():
            if args.print:
                print('move: ' + str(move + 1))
                env.print()
            return True
        if args.print:
            print('move' + str(move + 1))
            env.print()
    return False 

if __name__ == "__main__":
    
    argparser.set_up(parser)
    args = parser.parse_args()
    mrx_model = None
    agent_model = None
    
    # resets values to their default states if incorrectly inputed from console
    if args.gamma > 1 or args.gamma < 0:
        args.gamma = 0.975
    if args.epsilon > 1 or args.epsilon < 0:
        args.epsilon = 1
    
    # starts training process of Mr.X
    if args.train_model == 1:
        DQN_mrx.train_mrx(args.epochs, args.gamma, args.epsilon, args.batch, args.buffer, args.update_freq, args.field_size, args.agents, args.length, args.moves, args.sim_time, args.mrx_model, args.retrain)
    # starts training process of agent
    elif args.train_model == 2:
        DQN_agent.train_agent(args.epochs, args.gamma, args.epsilon, args.batch, args.buffer, args.update_freq, args.sim_time, args.field_size, args.moves, args.length, args.agent_model, args.retrain)
    # plays games
    else:
        grid_size = args.field_size *args.field_size * 2
        neurons = grid_size *2
        # loads Mr.X model if DQN Mr.X picked to play
        if args.mrx_type == 0:
            mrx_model = DQN_mrx.mrx_network(grid_size, neurons).to(device).float()
            mrx_model.load_state_dict(torch.load(args.mrx_model))
            mrx_model.eval()
        # loads agent model if DQN agents picked to play
        if args.agents_type == 0:
            agent_model = DQN_agent.agent_network(grid_size, neurons).to(device).float()
            agent_model.load_state_dict(torch.load(args.agent_model))
            agent_model.eval()
        
        agent_wins = 0
        
        start = datetime.now().time()
        
        for game in range(args.games):
            env.agents = list(env.agents)
            env.reset(args.field_size, args.agents, args.length)
            if args.print:
                print('game: ' + str(game+1))
                print('move: 0')
                env.print()
            if play_game(mrx_model, agent_model):
                agent_wins += 1
            
            print("agents won: " + str(agent_wins) + '/' + str(game + 1))
            
        print("total no. of games won by agents: " + str(agent_wins) + '/' + str(args.games))
        print('Time of start: ' + str(start))
        print('Time of end: ' + str(datetime.now().time()))