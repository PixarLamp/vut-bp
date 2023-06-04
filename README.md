# Launching the application

Bachelors thesis on topic "Deep learning methods for machine playing the Scotland Yard board game"<br>
implementation of DQN algorithm on simplified verion of Scotland Yard

## Author

Zuzana Hrkľová, xhrklo00<br>Brno University of Technology FIT, 2023

## Instalation

Necessities include python 3.0 or newer version and PyTorch

## Usage

```
main.py [-h] [-g GAMES] [-s {4,5,6,7,8,9,10}] [-m {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}] [--mrx-ls {3,4,5}] [-t SIM_TIME] [-a {1,2,3,4}] [--mrx-type {0,1,2,3,4}][--agents-type {0,1,2,3,4}] [--mrx_model MRX_MODEL] [--agent_model AGENT_MODEL] [--train-model {0,1,2}] [--retrain] [--epochs EPOCHS] [--gamma GAMMA] [-e EPSILON] [--batch BATCH] [--buffer BUFFER] [--update-freq UPDATE_FREQ] [--print]
```

# Optional Arguments 

```
-h, --help                                  Shows help message and exit`
-g GAMES, --games GAMES                     Number of games to be played (default 1000)
-s FIELD_SIZE, --field-size FIELD_SIZE      Size of field (default 5)
-m MOVES, --moves MOVES                     Number of moves in one game (default 15)
--mrx-ls LENGTH                             Number of moves when Mr. X is hidden (default 3)
-t SIM_TIME, --sim-time SIM_TIME            Simulation time used by Monte Carlo Tree Search (default 0.05s)
-a AGENTS, --agents AGENTS                  Number of agents (default 2)
--mrx-type MRX_TYPE                         Type of method used for Mr. X:
                                                0 - DQN (default)
                                                1 - Alfa-Beta
                                                2 - Monte-Carlo MG 2022
                                                3 - Monte-Carlo MS 2021
                                                4 - Random movement
--agents-type AGENTS_TYPE                   Type of method used for Mr. X:
                                                0 - DQN
                                                1 - Alfa-Beta (default)
                                                2 - Monte-Carlo MG 2022
                                                3 - Monte-Carlo MS 2021
                                                4 - Random movement
--mrx_model MRX_MODEL                       Trained model used for Mr.X movement (default ./trained_models/mrx_model.pt)                
--agent_model AGENT_MODEL                   Trained model used for agent movement (default ./trained_models/agents_model.pt)
--train-model TRAIN_MODEL                   Starts training DQN model:
                                                0 - No training (default)
                                                1 - Train Mr.X
                                                2 - Train agent
--retrain RETRAIN                           Retrains existing DQN model (default False)
--epochs EPOCHS                             Number of games played during training of DQN model (default 10000)
--gamma GAMMA                               Discount factor used during DQN training (default 0.975)
-e EPSILON, --epsilon                       Epsilon value used during DQN training (default 1)
--batch BATCH                               Size of batches sampled from replay memory during DQN training (default 512)
--buffer BUFFER                             Size of a replay memory used during DQN training (default 15000)
--update-freq UPDATE_FREQ                   Frequency with which target network gets updated with new weights during DQN training (default 100)
--print PRINT                               Prints game states during game play (default True)
```

## Self-implemented source files

* argparser.py
* main.py
* DQN\_agent.py
* DQN\_mrx.py
* player\_method.py

## Third-party source files

AT\_2020 - BP A. Tulušák 2020 VUT FIT
* alfa\_beta.py
* alfa\_beta\_mrx.py
* alfa\_beta\_node.py
* environment.py
* player\_input.py

MS\_2021 - BP M. Sova 2021 VUT FIT
* alfa\_beta\_wrapper.py
* mc\_environment.py
* monte\_carlo.py
* player.py

MC\_2022 - BP M. Gerža 2022 VUT FIT
* mc\_environment.py
* MG\_monte\_carlo.py