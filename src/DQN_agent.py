# author: Zuzana Hrkľová
# year: 2023
# file: DQN_agent.py
# description: Implementation of DQN algorithm for training and moving Scotland Yard's agents

import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import player_method

# Using of Michal Sova's mc_environment edited by Martin Gerža
from MC_2022 import mc_environment
env = mc_environment.Environment()

device = "cuda" if torch.cuda.is_available() else "cpu"

# neural network used to play agent's moves
class agent_network(nn.Module):
    def __init__(self, grid_size=50, neurons=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(grid_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 4),
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

# calculates probability of Mr.X's occurances on the board
def mrx_position_probability(prev_probability, mrx_visible, mrx_pos, agents, field_size):
    env.mrx_visible = mrx_visible
    env.mrx_pos = mrx_pos
    env.agents = agents
    
    probability = []
    
    if env.mrx_visible:
        probability.append([env.mrx_pos[0], env.mrx_pos[1], 1])
        return probability
    
    for agent in env.agents:
        for prob in prev_probability:
            if agent == [prob[0], prob[1]]:
                prev_probability.remove(prob)
    
    for act in prev_probability:
        next_x_moves = []
        if act[0] - 1 >= 0 and [act[0] - 1, act[1]] not in env.agents:
            next_x_moves.append([act[0] - 1, act[1]])
        if act[0] + 1 < field_size and [act[0] + 1, act[1]] not in env.agents:
            next_x_moves.append([act[0] + 1, act[1]])
        if act[1] - 1 >= 0 and [act[0], act[1] - 1] not in env.agents:
            next_x_moves.append([act[0], act[1] - 1])
        if act[1] + 1 < field_size and [act[0], act[1] + 1] not in env.agents:
            next_x_moves.append([act[0], act[1] + 1])
        
        for move in next_x_moves:
            added = False
            for tile in probability:
                if tile[0] == move[0] and tile[1] == move[1]:
                    tile[2] += act[2] / len(next_x_moves)
                    added = True
            if not added:
                probability.append([move[0], move[1], act[2] / len(next_x_moves)])
            added = False 
           
    return probability

# used to make first available move for agent when he tries to move out of board space or steps on other agent's position
def alternative_move(agent_pos):
    x = 0 
    while(x < 4):
        new_pos = [agent_pos[0] + actions[x][0], agent_pos[1] + actions[x][1]]
        if (0 <= new_pos[0] < env.field_size and 0 <= new_pos[1] < env.field_size):
            if new_pos not in env.agents:
                env.agents.append(new_pos)
                return True
        x += 1

# performs one move of DQN agent
def move_NN_agent(agent_pos, action, field_size, agents):
    env.agents = agents
    new_pos = [agent_pos[0] + action[0], agent_pos[1] + action[1]]

    if (0 <= new_pos[0] < field_size and 0 <= new_pos[1] < field_size):
        if new_pos not in env.agents:
            env.agents.append(new_pos)
            return True, env.agents
        else:
            alternative_move(agent_pos)
            return False, env.agents
    else:
        alternative_move(agent_pos)
        return False, env.agents

# reward function
def get_reward(game_status):
    if game_status == 1:
        return 10 #agent contributes to winning
    if game_status == 2:
        return -10 #agents loose
    if game_status == 3:
        return 0 
    if game_status == 4:
        return 1 #agent steps on a probable location of Mr.X

# creates grid needed to create input state of the neural network
def create_grid(mrx_probability, NN_agent, field_size, agents):
    env.field_size = field_size
    env.agents = agents
    grid = np.zeros((2, env.field_size, env.field_size))
    for prob in mrx_probability:
        grid[0][prob[0], prob[1]] = prob[2]
    for agent in env.agents:
        grid[1][agent[0], agent[1]] = 1
    if NN_agent in env.agents:
        grid[1][NN_agent[0], NN_agent[1]] = 11
    else:
        grid[1][NN_agent[0], NN_agent[1]] = 10
    return grid

# outputs current game state for agent to use as an input of neural network
def get_state(move, mrx_probability, mrx_type, sim_time, field_size, moves, length, model):
    # Mr.X performs an action
    valid_move, new_position = player_method.move_mrx(move, mrx_type, env.mrx_pos, env.agents, field_size, moves, env.last_seen_length, sim_time, model)
    env.mrx_pos = new_position
    #setup of Mr.X's visibility 
    if ((move + 1) % 3 == 0):
        env.mrx_visible = True
        env.mrx_last_seen = 1
    else:
        env.mrx_visible = False
        env.mrx_last_seen += 1
                    
    #calculates probability needed for input state of neural network
    mrx_probability = mrx_position_probability(mrx_probability, env.mrx_visible, env.mrx_pos, env.agents, field_size)         
    #take NN agent off the grid so the Alpha-Beta agent can make his move
    NN_agent_pos = env.agents.pop()
    #move the first agent with Alpha-Beta
    
    move_made, agents_new_pos = player_method.move_agents(move, 1, env.agents, field_size, mrx_probability, moves, 0, env.mrx_last_seen, length, env.mrx_ls_pos, env.mrx_pos, model)
    env.agents = agents_new_pos
    # creates state from gained information
    grid = create_grid(mrx_probability, NN_agent_pos, field_size, env.agents)
    grid_size = field_size*field_size*2 
    state = grid.reshape(1, grid_size)
    
    return state, mrx_probability, NN_agent_pos, grid
# possible actions
actions = {
    0: [0, 1], #right
    1: [1, 0], #down
    2: [0, -1], #left
    3: [-1, 0], #up
}

def train_agent(epochs, gamma, epsilon, batchSize, buffer, update_freq, sim_time, field_size, moves, length, agents_model, retrain):
    print(f"Using {device} device")
    #neural network size setup
    grid_size = field_size*field_size*2 
    neurons = grid_size*2
    
    if retrain:
        # loads an existing model if retrain option chosen
        model = agent_network(grid_size, neurons).to(device).float()
        model.load_state_dict(torch.load(agents_model))
        model.eval()
        policy_net = copy.deepcopy(model)
    else:
        # creates new model if none was loaded
        policy_net = agent_network(grid_size, neurons).to(device).float()
    target_net = copy.deepcopy(policy_net)
    
    PATH = "agent_model.pt"
    BEST_PATH = 'agent_model_lowest_lr.pt'
    
    loss_fn = torch.nn.HuberLoss()
    
    replay = []
    loss_memory = []
    reward_memory = []
    mrx_probability = []
    
    index = 0
    won = 0
    update_target_net = 0
    best_loss = 0
    strat_t = datetime.now().time()
    
    for epoch in range(epochs):
        # one full game
        print("epoch no : " + str(epoch))
        env.reset(field_size, 2, length)

        move = 0
        reward = 0
        status = 1
        mrx_probability = []
        mrx_type = random.randint(1,2)
        
        while (status == 1):
            m_reward = 0
            if (move == 0):
                # getting very first state
                state, mrx_probability, NN_agent_pos, grid = get_state(move, mrx_probability, mrx_type, sim_time, field_size, moves, length, model)
                state1 = torch.tensor(state)
                state1 = state1.to(torch.device(device))
            
            qval = policy_net(state1.float()).data.cpu().numpy()
            #exploration vs. exploitation using epsilon-greedy strategy
            if (random.random() < epsilon):
                action = np.random.randint(0,4)
            else:
                action = np.argmax(qval)
            
            chosen_action = actions[action]
            
            terminal_state = False
            # make a chosen move and get reward
            if not move_NN_agent(NN_agent_pos, chosen_action, env.field_size, env.agents):
                m_reward += get_reward(3)
                reward += m_reward
            
            if env.mrx_visible:
                if [env.mrx_pos[0], env.mrx_pos[1] + 1] == env.agents[-1]:
                    m_reward += get_reward(4)
                    reward += m_reward
                if [env.mrx_pos[0], env.mrx_pos[1] - 1] == env.agents[-1]:
                    m_reward += get_reward(4)
                    reward += m_reward
                if [env.mrx_pos[0] + 1, env.mrx_pos[1]] == env.agents[-1]:
                    m_reward += get_reward(4)
                    reward += m_reward
                if [env.mrx_pos[0] - 1, env.mrx_pos[1]] == env.agents[-1]:
                    m_reward += get_reward(4)
                    reward += m_reward
            if grid[0][env.agents[-1][0], env.agents[-1][1]] > 0:
                m_reward += get_reward(4)
                reward += m_reward
            
            print('move: '  + str(move))
            env.print()
            print('reward: ' + str(reward))
            
            # agents won
            if env.win_condition():
                reward += get_reward(1)

                terminal_state = True
                won += 1
                reward_memory.append(reward)
            # last move, mr X won
            elif move == moves - 1:
                m_reward += get_reward(2)
                reward += m_reward
                terminal_state = True
                reward_memory.append(reward)
            
            # play mr x move in advance to find out if he has any possible moves
            save_x_pos = env.mrx_pos
            if terminal_state == False:
                valid_move, new_position = player_method.move_mrx(move, mrx_type, env.mrx_pos, env.agents, field_size, moves, env.last_seen_length, sim_time, model)
                env.mrx_pos = new_position
                if not valid_move:
                    m_reward += get_reward(1)
                    reward += m_reward
                    terminal_state = True
                    won += 1
                    reward_memory.append(reward)
            #reverse state of mr X back
            env.mrx_pos = save_x_pos
            
            state, mrx_probability, NN_agent_pos, grid = get_state(move+1, mrx_probability, mrx_type, sim_time, field_size, moves, length, model)

            #retyping input states
            state2 = torch.tensor(state)
            state2 = state2.to(torch.device(device))
            
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
                
                # update policy network
                policy_net.optimizer.zero_grad()
                loss = loss_fn(q_pred, y_j)
                loss.backward()
                loss_memory.append(loss.item())
                policy_net.optimizer.step()
                
                if update_target_net == update_freq:
                    # update target network
                    target_net = copy.deepcopy(policy_net)
                    update_target_net = 0
                else:
                    update_target_net += 1
                # save model when reaches lowest learning rate so far
                if epoch+1 == batchSize:
                    best_loss = loss
                if loss < best_loss:
                    torch.save(policy_net.state_dict(), BEST_PATH)
                
            move += 1
            if terminal_state:
                # game ends
                status = 0
            
        print('won ' + str(won) + '/' + str(epoch + 1))
        if epsilon > 0.01:
            epsilon -= 1/epochs  
            
    loss_memory = np.array(loss_memory)
    torch.save(policy_net.state_dict(), PATH)
    
    print('Time of start : ' + str(strat_t))
    print('Time of end: ' + str(datetime.now().time()))
    print('games won : ' + str(won))
   
    # saves learning rate graph from the performed training
    plt.plot(loss_memory)
    plt.ylabel('loss')
    plt.xlabel('steps')
    plt.title("Learning rate")
    plt.savefig('agent_learning_rate.png')
    plt.show()
    
    # saves reward graph from the performed training
    plt.plot(reward_memory)
    plt.ylabel('reward')
    plt.xlabel('epoch')
    plt.title('Total reward')
    plt.savefig('agent_rewards.png')
    plt.show()