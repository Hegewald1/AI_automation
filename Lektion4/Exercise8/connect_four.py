# PacktPublishing
# Artificial Intelligence with Python
# Chapter 13 .

# This is a variant of the Connect Four recipe given in the easyAI library

import numpy as np
from easyAI import AI_Player, Negamax, SSS
from copy import deepcopy

class TwoPlayersGame:
    """
    Base class for... wait for it... two-players games !

    To define a new game, make a subclass of TwoPlayersGame, and define
    the following methods:

    - ``__init__(self, players, ...)`` : initialization of the game
    - ``possible_moves(self)`` : returns of all moves allowed
    - ``make_move(self, move)``: transforms the game according to the move
    - ``is_over(self)``: check whether the game has ended

    The following methods are optional:

    - ``show(self)`` : prints/displays the game
    - ``scoring``: gives a score to the current game (for the AI)
    - ``unmake_move(self, move)``: how to unmake a move (speeds up the AI)
    - ``ttentry(self)``: returns a string/tuple describing the game.

    The __init__ method *must* do the following actions:

    - Store ``players`` (which must be a list of two Players) into
      self.players
    - Tell which player plays first with ``self.nplayer = 1 # or 2``

    When defining ``possible_moves``, you must keep in mind that you
    are in the scope of the *current player*. More precisely, a
    subclass of TwoPlayersGame has the following attributes that
    indicate whose turn it is. These methods can be used but should not
    be overwritten:

    - ``self.player`` : the current player (e.g. ``Human_Player``)
    - ``self.opponent`` : the current Player's opponent (Player).
    - ``self.nplayer``: the number (1 or 2) of the current player.
    - ``self.nopponent``: the number (1 or 2) of the opponent.
    - ``self.nmove``: How many moves have been played so far ?

    For more, see the examples in the dedicated folder.

    Examples:
    ----------

    ::

        from easyAI import TwoPlayersGame, Human_Player

        class Sticks( TwoPlayersGame ):
            ''' In turn, the players remove one, two or three sticks from
                a pile. The player who removes the last stick loses '''

            def __init__(self, players):
                self.players = players
                self.pile = 20 # start with 20 sticks
                self.nplayer = 1 # player 1 starts
            def possible_moves(self): return ['1','2','3']
            def make_move(self,move): self.pile -= int(move)
            def is_over(self): return self.pile <= 0


        game = Sticks( [Human_Player(), Human_Player() ] )
        game.play()


    """

    def play(self, nmoves=1000, verbose=True):

        history = []

        if verbose:
            self.show()

        for self.nmove in range(1, nmoves + 1):

            if self.is_over():
                break

            move = self.player.ask_move(self)
            history.append((deepcopy(self), move))
            self.make_move(move)

            if verbose:
                print("\nMove #%d: player %d plays %s :" % (
                    self.nmove, self.nplayer, str(move)))
                self.show()

            self.switch_player()

        history.append(deepcopy(self))

        return history

    @property
    def nopponent(self):
        return 2 if (self.nplayer == 1) else 1

    @property
    def player(self):
        return self.players[self.nplayer - 1]

    @property
    def opponent(self):
        return self.players[self.nopponent - 1]

    def switch_player(self):
        self.nplayer = self.nopponent

    def copy(self):
        return deepcopy(self)


class GameController(TwoPlayersGame):
    def __init__(self, players, board = None):
        # Define the players
        self.players = players

        # Define the configuration of the board
        self.board = board if (board != None) else (
            np.array([[0 for i in range(7)] for j in range(6)]))

        # Define who starts the game
        self.nplayer = 1

        # Define the positions
        self.pos_dir = np.array([[[i, 0], [0, 1]] for i in range(6)] +
                   [[[0, i], [1, 0]] for i in range(7)] +
                   [[[i, 0], [1, 1]] for i in range(1, 3)] +
                   [[[0, i], [1, 1]] for i in range(4)] +
                   [[[i, 6], [1, -1]] for i in range(1, 3)] +
                   [[[0, i], [1, -1]] for i in range(3, 7)])

    # Define possible moves
    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    # Define how to make the move
    def make_move(self, column):
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.nplayer

    # Show the current status
    def show(self):
        print('\n' + '\n'.join(
                ['0 1 2 3 4 5 6', 13 * '-'] +
                [' '.join([['.', 'O', 'X'][self.board[5 - j][i]]
                for i in range(7)]) for j in range(6)]))

    # Define what a loss_condition looks like
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
                if self.board[pos[0], pos[1]] == self.nopponent:
                    streak += 1
                    if streak == 4:
                        return True
                else:
                    streak = 0

                pos = pos + direction

        return False

    # Check if the game is over
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()

    # Compute the score
    def scoring(self):
        return -100 if self.loss_condition() else 0

if __name__ == '__main__':
    # Define the algorithms that will be used
    algo_neg = Negamax(5)
    algo_sss = SSS(5)

    # Start the game
    game = GameController([AI_Player(algo_neg), AI_Player(algo_sss)])
    game.play()

    # Print the result
    if game.loss_condition():
        print('\nPlayer', game.nopponent, 'wins.')
    else:
        print("\nIt's a draw.")