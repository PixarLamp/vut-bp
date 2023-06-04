# author: Zuzana Hrkľová
# year: 2023
# file: DQN_mrx.py
# description: Implementation of DQN algorithm for training and moving Scotland Yard's Mr.X

import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import DQN_agent
import player_method

# Using of Michal Sova's mc_environment edited by Martin Gerža
from MC_2022 import mc_environment

env = mc_environment.Environment()
 
device = "cuda" if torch.cuda.is_available() else "cpu"

# neural network used to play Mr.X
class mrx_network(nn.Module):
    def __init__(self, grid_size=50, neurons=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(grid_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, grid_size),
            nn.ReLU(),
            nn.Linear(grid_size, 4),
        )
        self.initialize_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.model(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

# reward function
def get_reward(game_status):
    if game_status == 1:
        return 1 # survived move 
    if game_status == 2:
        return -5 # game lost
    
# outputs state used as input to neural network 
def get_state(field_size, mrx_pos, agents):
    env.field_size = field_size
    env.mrx_pos = mrx_pos
    env.agents = agents
    grid = np.zeros((2, env.field_size, env.field_size,))
    grid[0][env.mrx_pos[0], env.mrx_pos[1]] = 1
    for agent in env.agents:
        grid[1][agent[0], agent[1]] = 1
    
    grid_size = field_size*field_size*2  
    state = grid.reshape(1, grid_size)
    return state

# performes one move of Mr.X
def move_mrx(action, field_size, mrx_pos, agents):
    env.mrx_pos = mrx_pos
    env.agents = agents
    new_pos = [env.mrx_pos[0] + action[0], env.mrx_pos[1] + action[1]]
    
    if (0 <= new_pos[0] < field_size and 0 <= new_pos[1] < field_size):
        env.mrx_pos = new_pos
        if new_pos not in env.agents:
            return 1, env.mrx_pos # made a good move
        else:
            return 2, env.mrx_pos # lost because he stepped on agent by itself
    else:
        x = 0 
        while(x < 4):
            new_pos = [env.mrx_pos[0] + DQN_agent.actions[x][0], env.mrx_pos[1] + DQN_agent.actions[x][1]]
            if (0 <= new_pos[0] < field_size and 0 <= new_pos[1] < field_size):
                env.mrx_pos = new_pos
                if new_pos not in env.agents:
                    return 3, env.mrx_pos # survived the move but went out of grid 
                else:
                    return 4, env.mrx_pos # lost after going out of grid
            x += 1

# trains Mr. X model
def train_mrx(epochs, gamma, epsilon, batchSize, buffer, update_freq, field_size, agents, length, moves, sim_time, mrx_model, retrain):
    print(f"Using {device} device")
    # parameters needed to set up size of neural network
    grid_size = field_size * field_size * 2
    neurons = grid_size * 2
    
    # loads an existing model if retrain option chosen
    if retrain:
        model = mrx_network(grid_size, neurons).to(device).float()
        model.load_state_dict(torch.load(mrx_model))
        model.eval()
        policy_net = copy.deepcopy(model)
    else:
        # creates new model if none was loaded
        policy_net = mrx_network(grid_size, neurons).to(device).float()
    target_net = copy.deepcopy(policy_net)
    
    PATH = "trained_mrx_model.pt"
    
    loss_fn = torch.nn.HuberLoss()
    
    replay = []
    loss_memory = []
    reward_memory = []    

    index = 0
    won = 0
    update_target_net = 0
    start_t = datetime.now().time()
    print('Time of start : ' + str(start_t))
    
    for epoch in range(epochs):
        # one full game
        print("epoch no : " + str(epoch))
        env.agents = list(env.agents)
        env.reset(field_size, agents, length)
        move = 0
        reward = 0
        status = 1
        agent_type = random.randint(1,2)
        
        while (status == 1):
            m_reward = 0
            if (move == 0):
                state = get_state(field_size, env.mrx_pos, env.agents)
                state1 = torch.tensor(state)
                state1 = state1.to(torch.device(device))
            
            qval = policy_net(state1.float()).data.cpu().numpy()
            #exploration vs. exploitation using epsilon-greedy strategy
            if (random.random() < epsilon):
                action = np.random.randint(0,4)
            else:
                action = np.argmax(qval)
            
            chosen_action = DQN_agent.actions[action]
            
            terminal_state = False
            # make a chosen move
            move_stat, new_pos = move_mrx(chosen_action, field_size, env.mrx_pos, env.agents)
            
            # visibility setup
            if ((move + 1) % 3 == 0):
                env.mrx_visible = True
                env.mrx_last_seen = 1
                env.mrx_ls_pos = env.mrx_pos
            else:
                env.mrx_visible = False
                env.mrx_last_seen += 1
                
            if env.win_condition():
                # Mr.X lost
                m_reward += get_reward(2)
                reward += m_reward
                
                terminal_state = True
                won += 1
                reward_memory.append(reward)
            else: 
                move_made, agents_new_pos = player_method.move_agents(move, agent_type, env.agents, field_size, [], moves, sim_time, env.mrx_last_seen, length, env.mrx_ls_pos, env.mrx_pos, model)
                env.agents = agents_new_pos

                if env.win_condition():
                    # Mr.X lost 
                    m_reward += get_reward(2)
                    reward += m_reward
                    
                    terminal_state = True
                    won += 1
                    reward_memory.append(reward)
                elif move == moves - 1:
                    terminal_state = True
                    reward_memory.append(reward)
                if move_stat == 1 and not env.win_condition():
                    # Mr.X won
                    m_reward += get_reward(1)
                    reward += m_reward
            print('move: '  + str(move))
            env.print()
            print(reward)
            # calculates next state needed in DQN algorithm 
            state = get_state(field_size, env.mrx_pos, env.agents)

            state2 = torch.tensor(state)
            state2 = state2.to(torch.device(device))
            # gathers experience gained from the last performed move
            exp = (state1, action, m_reward, state2, terminal_state)
            
            state1 = state2
            
            #experience replay storage
            if (len(replay) < buffer):
                replay.append(exp)
            else:
                if index < buffer - 1:
                    index += 1
                else:
                    index = 0
                replay[index] = exp

            if len(replay) > batchSize:
                # creates batch from replay memory 
                minibatch = random.sample(replay, batchSize)
                
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch]).to(torch.device(device))
                reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch]).to(torch.device(device))
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch]).to(torch.device(device))
                
                q_pred = policy_net.forward(state1_batch.float()).gather(1, action_batch.view(-1, 1))
                q_target = target_net.forward(state2_batch.float()).max(dim=1).values
                
                for idx in range(len(done_batch)):
                    if done_batch[idx] == True:
                        q_target[idx] = 0.0
                
                y_j = reward_batch + (gamma * q_target)
                y_j = y_j.view(-1, 1)
                
                #policy network update
                policy_net.optimizer.zero_grad()
                loss = loss_fn(q_pred, y_j)  
                loss.backward()
                loss_memory.append(loss.item())
                policy_net.optimizer.step()
                
                if update_target_net == update_freq:
                    # target network update
                    target_net = copy.deepcopy(policy_net)
                    update_target_net = 0
                else:
                    update_target_net += 1
                
            move += 1
            if terminal_state:
                # game ends
                status = 0
            
        print('reward = ' + str(reward))
        print('won ' + str(won) + '/' + str(epoch + 1))
        if epsilon > 0.01:
            epsilon -= 1/epochs    
            
    loss_memory = np.array(loss_memory)
    torch.save(policy_net.state_dict(), PATH)
    
    print('Time of start : ' + str(start_t))
    print('Time of end: ' + str(datetime.now().time()))
    print('games won : ' + str(won))
   
    # saves learning rate graph from the performed training
    plt.plot(loss_memory)
    plt.ylabel('loss')
    plt.xlabel('steps')
    plt.title("Learning rate")
    plt.savefig('mrx_learning_rate.png')
    plt.close()
    
    # saves reward graph from the performed training
    plt.plot(reward_memory)
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.title('Total reward')
    plt.savefig('mrx_rewards.png')
    plt.close()   