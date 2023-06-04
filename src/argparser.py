# author: Zuzana Hrkľová
# year: 2023
# file: argparser.py
# description: Implementation of parser to set up console input

def set_up(parser):
     
     parser.add_argument("-g", "--games", default=1000, type=int, help="Number of games to be played (default 1000)")
     parser.add_argument("-s", "--field-size", default=5, type=int, choices=range(4, 11), help="Size of field (default 5)")
     parser.add_argument("-m", "--moves", default=15, type=int, choices=range(5, 25), help="Number of moves in one game (default 15)")
     parser.add_argument("--mrx-ls", dest="length", default=3, type=int, choices=range(3, 6), help="Number of moves when Mr. X is hidden (default 3)")
    
     parser.add_argument("-t", "--sim-time", default=0.05, type=float, help="Simulation time used by Monte Carlo Tree Search (default 0.05s)")

     parser.add_argument("-a", "--agents", default=2, type=int, choices=range(1, 5), help="Number of agents (default 2)")

     parser.add_argument("--mrx-type", default=0, type=int, choices=range(0, 5),
                        help="Type of method used for Mr. X:\n"
                             "0 - DQN (default)\n"
                             "1 - Alfa-Beta\n"
                             "2 - Monte-Carlo MG 2022\n"
                             "3 - Monte-Carlo MS 2021\n"
                             "4 - Random movement\n")
     parser.add_argument("--agents-type", default=1, type=int, choices=range(0, 5),
                        help="Type of used method for agents:\n"
                             "0 - DQN\n"
                             "1 - Alfa-Beta (default)\n"
                             "2 - Monte-Carlo MG 2022\n"
                             "3 - Monte-Carlo MS 2021\n"
                             "4 - Random movement")
    
     parser.add_argument("--mrx_model", default="./trained_models/mrx_model.pt", type=str, help="Trained model used for Mr.X movement (default ./trained_models/mrx_model.pt)")
     parser.add_argument("--agent_model", default="./trained_models/agents_model.pt", type=str, help="Trained model used for agent movement (default ./trained_models/agents_model.pt)")
     
     parser.add_argument("--train-model", default="0", type=int, choices=range(0, 3), 
                         help="Starts training DQN model:\n"
                              "0 - No training (default)\n"
                              "1 - Train Mr.X\n"
                              "2 - Train agent")
     parser.add_argument("--retrain", default="False", action='store_true', help="Retrains existing DQN model (default False)")
     
     parser.add_argument("--epochs", default="10000", type=int, help="Number of games played during training of DQN model (default 10000)")
     parser.add_argument("--gamma", default=" 0.975", type=float, help="Discount factor used during DQN training (default 0.975)")
     parser.add_argument("-e", "--epsilon", default="1", type=float, help="Epsilon value used during DQN training (default 1)")
     parser.add_argument("--batch", default="512", type=int, help="Size of batches sampled from replay memory during DQN training (default 512)")
     parser.add_argument("--buffer", default="15000", type=int, help="Size of a replay memory used during DQN training (default 15000)")
     parser.add_argument("--update-freq", default="100", type=int, help="Frequency with which target network gets updated with new weights during DQN training (default 100)")

     parser.add_argument("--print", default=True, action='store_false', help="Prints game states during game play (default True)")

     return parser