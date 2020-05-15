#%%
import AI
from TicTacToe import TicTacToe, TicTacToePlayer
import numpy as np

import time

game_shape = (3, 3)

# Construct network
n_inputs = game_shape[0]*game_shape[1]
layer_1 = 16
layer_2 = 32
layer_3 = 32
n_outputs = n_inputs
layers = [AI.FullyConnectedLayer(n_inputs, layer_1, 'relu'),
            AI.FullyConnectedLayer(layer_1, layer_2, 'relu'),
            AI.FullyConnectedLayer(layer_2, layer_3, 'relu'),
            AI.FullyConnectedLayer(layer_3, n_outputs, 'softmax')]
network = AI.NeuralNetwork(layers)
player_1 = TicTacToePlayer('O', network)

player_2 = TicTacToePlayer('X')

game = TicTacToe((player_1, player_2), game_shape)

# Create function for optimizer to play tic tac toe
def tictactoe_evaluator(individual, player=player_1, opponent=player_2):
    players = [player, opponent]
    # Assign optimizer parameters to network
    player.set_network(individual)

    # Reset game board
    game.reset()

    n_turns = 0
    while player.wins < 3 and n_turns < 100:
        game.ai_move(player)

        # Place a random marker
        places_left = np.where(game.board_ai == 0)
        rand_spot = np.random.randint(len(places_left[0]))
        game.place((places_left[0][rand_spot], places_left[0][rand_spot]), opponent)

        n_turns += 1
    
    # Reset player
    player.wins = 0
    return 1/n_turns

# Initialize optimizer
population_size = 100
n_generations = 100
tournament_selection_prob = 0.7
optimizer_size = n_inputs * layer_1 + layer_1 +\
                 layer_1 * layer_2 + layer_2 +\
                 layer_2 * layer_3 + layer_3 +\
                 layer_3 * n_outputs + n_outputs
mutation_prob = 1/optimizer_size

optimizer = AI.GeneticOptimizer(optimizer_size, population_size, tictactoe_evaluator, 
                    n_generations, tournament_selection_prob, mutation_prob)
t = time.time()
optimizer.optimize()
print(time.time() - t)

#%% Play the game
game.reset()
game.verbose = True
game.print_board()
player_1.set_network(optimizer.top_individual)

#%%
game.ai_move(player_1)

# %%
game.place((1,0), player_2)

