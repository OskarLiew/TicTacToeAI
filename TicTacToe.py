import numpy as np
class TicTacToe:
    def __init__(self, players, shape=(3,3), in_a_row = 3, verbose=False):
        player_ids = []
        for player in players:
            assert type(player) == TicTacToePlayer, 'Player not an object of player class'
            assert player.id not in player_ids, 'All players are the same'
            player_ids.append(player.id)
        self.players = players
        self.verbose = verbose

        assert max(shape) >= in_a_row, 'Board too small for win critereon'
        self.shape = shape
        self.in_a_row = 3
        self.board_ai = np.zeros(shape, dtype=int)
        self.board = np.empty(shape, dtype=str)
        self.board[self.board == ''] = ' '
        
    def place(self, position, player):
        if type(player) == TicTacToePlayer:
            marker = player.marker
        else:
            raise Exception('Error: Incorrect player type')

        row = position[0]
        col = position[1]

        if self.board[row, col] == ' ':
            self.board[row, col] = marker
            self.board_ai[row, col] = player.id

        self.print_board()
        self._check_board()

    def print_board(self):
        if self.verbose:
            board = self.board
            rows = self.shape[0]
            columns = self.shape[1]

            for row in range(rows):
                print('+---'*columns + '+')
                print(('| {} '*columns).format(*board[row, :]) + '|')

            print('+---'*columns + '+')

    def reset(self):
        #print("Resetting board")
        self.board = np.empty(self.shape, dtype=str)
        self.board[self.board == ''] = ' '
        self.board_ai = np.zeros(self.shape, dtype=int)

    def _check_board(self):
        # Check so board is not full
        if (self.board == ' ').sum() == 0:
                if self.verbose:
                    print('Stalemate!')
                self.reset()
                self.print_board()

        # Find diagonals in board
        diags = [self.board_ai[i:i+self.in_a_row, j:j+self.in_a_row].diagonal()\
                for i in range(self.board_ai.shape[0] - self.in_a_row + 1)\
                for j in range(self.board_ai.shape[1] - self.in_a_row + 1)]
        diags = diags + [np.rot90(self.board_ai[i:i+self.in_a_row, j:j+self.in_a_row]).diagonal()\
                for i in range(self.board_ai.shape[0] - self.in_a_row + 1)\
                for j in range(self.board_ai.shape[1] - self.in_a_row + 1)]

        player_wins = False
        for player in self.players:
            identity = player.id

            # Check horizontal and vertical lines
            tmp = self.board_ai == identity
            if any(tmp.sum(axis=0) >= self.in_a_row) | any(tmp.sum(axis=1) >= self.in_a_row):
                player_wins = True

            # Check diagonals
            for diag in diags:
                if (diag == identity).sum() >= self.in_a_row:
                    player_wins = True

            if player_wins:
                player.wins += 1
                if self.verbose:
                    print('Player', player.marker, 'wins!')
                self.reset()
                self.print_board()
                break
    def ai_move(self, player):
        board = self.board_ai.copy().flatten()
        board[board == player.id] = 1
        board[np.logical_and(board != 0, board != 1)] = -1

        # Calculate network output for board state
        output = player.network.output(board)
        sorted_output_idx = np.argsort(-output)

        # Find empty spots and place marker on best empty spot
        empty_spots = np.where(board == 0)[0]
        best_moves = np.in1d(sorted_output_idx, empty_spots)
        best_move = sorted_output_idx[best_moves][0]
        best_move = np.unravel_index(best_move, self.shape)
        self.place(best_move, player)

class TicTacToePlayer:
    def __init__(self, marker, network=None):
        self.id = np.random.randint(-2**31, 2**31)
        self.marker = marker
        self.wins = 0
        self.network = network

    def set_network(self, chromosome):
        assert self.network != None, 'Player does not have a network.'
        chromosome_pos = 0
        for layer in self.network.layers:
            n_weights = np.prod(layer.shape)
            n_thresholds = layer.shape[0]

            layer.set_thresholds(chromosome[chromosome_pos:chromosome_pos + n_thresholds])
            chromosome_pos += n_thresholds
            layer.set_weights(chromosome[chromosome_pos:chromosome_pos + n_weights].reshape(layer.shape))
            chromosome_pos += n_weights