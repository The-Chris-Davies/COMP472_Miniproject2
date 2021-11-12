# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import sys
import time
import numpy as np
from scipy.signal import convolve2d
from copy import deepcopy

class Agent:
    def __init__(self, player, heuristic=None, depth=None, time=float('inf'), **algo_args):
        '''intializes the Agent.
        algo - the function that chooses the agent's next move.
        player - the player that the Agent is playing as. Either 'X' or 'O'.
        heuristic - the heuristic function to use for minimax / alpha-beta AIs.
        depth - the maximum depth the agent can use to check the win conditions. Default is None, for no maximum depth.
        time - the maximum time that the agent can take to make a move. Default is None, for no maximum time.
        Any additional arguments are passed to the algorithm when called.'''
        self.p = player
        self.d = depth
        self.t = float(time)*0.9    #10% margin of error for time allotment
        self.args = algo_args
        self.heuristic = heuristic
    def get_move_info(self):
        '''returns a dict with information about the last move the agent performed.
        returns moveInfo - dict: hevals: number of heuristic evaluations
                                 heval_d: number of heuristic evaluations by depth
                                 d_avg: average evaluation depth
                                 r_d: average recursion depth'''
        moveInfo = {}
        moveInfo['hevals'] = sum(self.heval_d.values())
        moveInfo['heval_d'] = self.heval_d
        dsum=0
        for k in self.heval_d:
            dsum += k*self.heval_d[k]
        moveInfo['d_avg'] = dsum / max(moveInfo['hevals'], 1)
        moveInfo['r_d'] = self.r_sum / max(self.r_n,1)

        return moveInfo

    def heuristic_1(self, game):
        '''heuristic 1: a weighted sum of the number of k-in-a-rows that a player has.
        Calculated by counting the number of potential winstates with (2 to s) pieces in them.
        '''

        score = 0

        #convolve the individual board states
        convs = [game.convolve_winstates(game.current_state[self.p]),
                 game.convolve_winstates(game.current_state['X' if self.p == 'O' else 'O']),
                 game.convolve_winstates(game.current_state['B'])]

        # combines the convolved matricies to a usable state 
        # any entry in our convolution that also has an entry in the opponents, or in b, is blocked.
        for i in range(len(convs[0])):
            for j in range(2,s):
                # score: the number of unobstructed win conditions, weighted by 2^(length of the condition)
                # this feels like code golf and github copilot had a child, and it's ugly
                score += (np.count_nonzero(convs[0][i][convs[1][i] == 0 and convs[2][i] == 0] == j) - np.count_nonzero(convs[1][i][convs[0][i] == 0 and convs[2][i] == 0] == j))<<j
        return score

    def heuristic_2(self, game):
        '''heuristic 2: we take the convolved winstates for ourselves and subtract our opponent's, then weight them.
        faster than h_1, but less useful
        '''

        score = 0

        #convolve the individual board states
        convs = [game.convolve_winstates(game.current_state[self.p]),
                 game.convolve_winstates(game.current_state['X' if self.p == 'O' else 'O'])]

        # combines the convolved matricies to a usable state 
        # any entry in our convolution that also has an entry in the opponents, or in b, is blocked.
        for i in range(len(convs[0])):
            for j in range(2,s):
                #score: the number of unobstructed win conditions * the number of pieces in that condition (- the same for the opponent)
                score += (np.count_nonzero(convs[0][i] == j) - np.count_nonzero(convs[1][i] == j))<<2
        return score

    def human(self, game):
        '''A human player. Gets player input from the command line.'''

        # initialize moveInfo
        self.heval_d = {0:1}
        self.r_sum = 0
        self.r_n = 1

        while True:
            move = input(F'Player {game.player_turn}, enter your move:').split()
            px = ord(move[0].lower()) - ord('a')
            py = int(move[1])

            if game.is_valid(px, py):
                return (px,py)
            else:
                print('The move is not valid! Try again.')

    def _heuristicless_minimax(self, game, max, d, endt):
        # Naively searches the full state space - no heuristic used
        # Do not call this function, call the non-underscore version instead.
        # Possible values are:
        # -1 - win
        # 0  - a tie
        # 1  - loss
        # We're initially setting it to 3 or -3 as worse than the worst case:
        # (and it is set to 2/-2 if depth is reached)


        # initialize metrics
        self.r_sum += d
        self.r_n += 1

        value = 3
        if max:
            value = -3
        x = None
        y = None
        # if game is ended, return win state
        result = game.is_end()
        if result == None:
            pass
        elif result == self.p:
            return (1, x, y)
        elif result == '.':
            return (0, x, y)
        else:
            return (-1, x, y)

        for i in range(0, game.n):
            for j in range(0, game.n):
                if game.is_valid(i,j):
                    # if current depth is equal to max depth or time is out, take the first valid move
                    if(d == self.d or time.time() > endt):
                        if not (value == 1 or value == -1):
                            value = -2 if max else 2
                            x = i
                            y = j

                    #print(F'looking at {i},{j} at depth {d}')

                    elif max:
                        #set position to AI's color
                        game.current_state[self.p][i][j] = 1
                        (v, _, _) = self._heuristicless_minimax(game, False, d+1, endt)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        #set position to Opponent's color
                        game.current_state['X' if self.p == 'O' else 'O'][i][j] = 1
                        (v, _, _) = self._heuristicless_minimax(game, True, d+1, endt)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    #undo the move at the end
                    game.undo_move(i,j)
        return (value, x, y)

    def heuristicless_minimax(self, game):
        '''heuristicless_minimax: naively searches the entire game state space to find a move
        calls _heuristicless_minimax, and returns the location of the best move.
        careful, it's SLOW! (shocking)'''
        endt = time.time() + self.t

        # initialize moveInfo
        self.heval_d = {}
        self.r_sum = 0
        self.r_n = 0

        (_,px,py) = self._heuristicless_minimax(game, True, 0, endt)
        return (px,py)

    def _heuristicless_alphabeta(self, game, alpha, beta, max, d, endt):
        # Do not call this function, call the non-underscore version instead.
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 3 or -3 as worse than the worst case:
        # (and it is set to 2/-2 if depth is reached)

        # initialize metrics
        self.r_sum += d
        self.r_n += 1

        value = 3
        if max:
            value = -3
        x = None
        y = None

        # if game is ended, return win state
        result = game.is_end()
        if result == None:
            pass
        elif result == self.p:
            return (1, x, y)
        elif result == '.':
            return (0, x, y)
        else:
            return (-1, x, y)

        for i in range(0, game.n):
            for j in range(0, game.n):
                if game.is_valid(i,j):
                    # if current depth is equal to max depth or time is out, take the first valid move
                    if(d == self.d or time.time() > endt):
                        print('d reached')
                        if not (value == 1 or value == -1):
                            value = -2 if max else 2
                            x = i
                            y = j

                    elif max:
                        game.current_state[self.p][i][j] = 1
                        (v, _, _) = self._heuristicless_alphabeta(game, alpha, beta, False, d+1, endt)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        game.current_state['X' if self.p == 'O' else 'O'][i][j] = 1
                        (v, _, _) = self._heuristicless_alphabeta(game, alpha, beta, True, d+1, endt)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    game.undo_move(i,j)

                    if max:
                        if value >= beta:
                            return (value, x, y)
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return (value, x, y)
                        if value < beta:
                            beta = value
        print(F"{' '*d} xy=({x},{y})")
        return (value, x, y)

    def heuristicless_alphabeta(self, game):
        '''heuristicless alphabeta: searches the game space using alpha-beta pruning, 
        and returns the location of the best move.'''

        endt = time.time()+self.t

        # initialize moveInfo
        self.heval_d = {}
        self.r_sum = 0
        self.r_n = 0

        (_,px,py) = self._heuristicless_alphabeta(game, True, 3, -3, 0, endt)
        return (px,py)


class Game:
    def __init__(self, n, s, blocs, t):
        '''initializes the game object.
        n - int - the dimensions of the game board (nxn)
        s - int - the number of pieces in a row needed to win
        blocs - list-like that contains the position of blocs to place on the grid, in the form
                [(x1,y1),(x2,y2),...,(xb,yb)]
        '''
        self.n = n
        self.s = s
        self.blocs = blocs
        self.t = t
        self.initialize_game()
        self.construct_winstates()

    def initialize_game(self):
        '''constructs the game representation, places blocs, and sets the player's turn.
        The state is a dict composed of two arrays: one for player X and one for player O.
        Each of these arrays holds empty spaces as 0 and occupied spaces as 1.
        int8 is used as the datatype, as bool arrays cannot be convolved.'''

        self.current_state = {'X': np.zeros((self.n,self.n), dtype=np.int8),
                              'O':np.zeros((self.n,self.n), dtype=np.int8),
                              'B':np.zeros((self.n,self.n), dtype=np.int8)}

        self.place_blocs()

        # Player X always plays first
        self.player_turn = 'X'

    def place_blocs(self):
        '''places the blocs into the game state at the specified location'''
        for bloc in self.blocs:
            self.current_state['B'][bloc[0],bloc[1]] = 1

    def construct_winstates(self):
        '''generates a list of all the possible winstates (horizontal, vertical, ascending, descending)
        for use determining win condition and in heuristic calculation.'''
        self.winstates = (
            np.ones((self.s,1), dtype=np.int8),    #horizontal
            np.ones((1,self.s), dtype=np.int8),    #vertical
            np.identity(self.s, dtype=np.int8),    #descending
            np.flipud(np.identity(self.s, dtype=bool)))  #ascending

    def draw_board(self):
        '''Returns a string with a graphical representation of the board state, using '+' as a block.'''
        board = ''
        # print column index (alpha)
        board += ' |' + ''.join([chr(let) for let in range(ord('A'), ord('A')+self.n)]) + '\n'
        board += '-+' + '-'*self.n + '\n'

        for y in range(0, self.n):
            # print row index (numeric)
            board += str(y) + '|'

            for x in range(0, self.n):
                if self.current_state['X'][x,y]:
                    board += 'X'
                elif self.current_state['O'][x,y]:
                    board += 'O'
                elif self.current_state['B'][x,y]:
                    board += '+'
                else:
                    board += '.'
            board += '\n'
        board += '\n'

        return board

    def is_valid(self, px, py):
        '''Checks whether the move (px,py) is a valid placement'''
        if px < 0 or px > self.n or py < 0 or py > self.n:
            return False
        elif self.current_state['X'][px,py] or self.current_state['O'][px,py] or self.current_state['B'][px,py]:
            return False
        else:
            return True

    def is_end(self):
        '''checks if a player has won, using convolution on the player move arrays.
        Returns 'X' if x has won, 'O' if o has won, '.' if a draw has occurred, and None otherwise.
        Uses convolution across the board state arrays to determine a win.
        Note that I didn't see that the max board size was 10 until much later, so this is probably seriously overbuilt. I figured it would go up to like 1000 or something ridiculous to really stress our algorithms.'''

        #check if X has won
        x_wins = self.convolve_winstates(self.current_state['X'])
        #if x_wins contains any entry with value s, then x has won.
        for state in x_wins:
            if self.s in state: return 'X'

        #check if O as won
        o_wins = self.convolve_winstates(self.current_state['O'])
        for state in o_wins:
            if self.s in state: return 'O'

        #check if board is full(tie)
        if(self.check_board_full()):
            return '.'

        return None

    def convolve_winstates(self, state):
        '''Convolves each of the possible win states (horizontal, vertical, ascending, descending)
        with the given player's board state and returns a list of the results. Used for determining win conditions, and for heuristic calculation.'''
        convs = []
        #perform the convolution on every winstate
        for winstate in self.winstates:
            convs.append(convolve2d(state, winstate, mode='valid'))
        return convs

    def check_board_full(self):
        '''checks if the current state has any possible moves'''
        # if the sum of all 3 state arrays has any zeros, there is still a possible move.
        if ((self.current_state['X'] + self.current_state['O'] + self.current_state['B']) == 0).any():
            return False
        return True

    def make_move(self, x,y, p):
        '''Makes the move at the given location for player p. Returns false if the move is invalid.'''
        if not self.is_valid(x,y):
            print("i should not be here")
            return False
        self.current_state[p][x,y] = 1
        return True

    def undo_move(self, x,y):
        '''Undoes a move at the given position. Used for simulating moves (& cheating)'''
        self.current_state['X'][x,y] = 0
        self.current_state['O'][x,y] = 0

    def check_end(self):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result != None:
            if self.result == 'X':
                print('The winner is X!')
            elif self.result == 'O':
                print('The winner is O!')
            elif self.result == '.':
                print("It's a tie!")
            self.initialize_game()
        return self.result

    def switch_player(self):
        if self.player_turn == 'X':
            self.player_turn = 'O'
        elif self.player_turn == 'O':
            self.player_turn = 'X'
        return self.player_turn

    def save_log(self, trace):
        '''saves a game trace (stored as a list)'''
        f=open(F"gameTrace-{self.n}{len(self.blocs)}{self.s}{self.t}", 'w')
        f.write('\n'.join(trace))
        f.close()

    def play(self,agent_x=Agent(Agent.human, 'X'),agent_o=Agent(Agent.human, 'O'), screenOut=True, log=False):
        '''runs the game with the specified agents.
        screenOut - bool - whether to print output to the screen
        log - bool - whether to save the game trace to a file'''
        agents = {'X': agent_x,
                  'O': agent_o}

        #initialize trace
        trace = []
        trace.append(F"n={self.n} b={len(self.blocs)} s={self.s} t={self.t}")
        trace.append(F"blocs={self.blocs}")
        trace.append(F"Player 1: {'Human' if agent_x.move == agent_x.human else 'AI'} d={agent_x.d} {agent_x.move.__name__}")
        trace.append(F"Player 2: {'Human' if agent_o.move == agent_o.human else 'AI'} d={agent_o.d} {agent_o.move.__name__}")

        while True:
            board = self.draw_board()
            trace.append(board)
            if screenOut: print(trace[-1])

            if self.check_end():
                if(log): self.save_log(trace)
                return

            start = time.time()
            #call the agent's move function
            move = agents[self.player_turn].move(self)
            #if the move was illegal, the agent loses
            if(not self.make_move(*move, self.player_turn)):
                trace.append(F"agent {self.player_turn} loses: made an illegal move.")
                if screenOut: print(trace[-1])
                return
            end = time.time()

            trace.append(F'Player {self.player_turn} plays: {chr(ord("A")+move[0])} {move[1]}')
            if(screenOut): print(trace[-1])

            #get statistics etc for the last move
            moveInfo = agents[self.player_turn].get_move_info()

            trace.append(F"i\tEvaluation time: {round(end - start, 7)}s")
            if(screenOut): print(trace[-1])
            trace.append(F"ii\tHeuristic evaluations: {moveInfo['hevals']}")
            if(screenOut): print(trace[-1])
            trace.append(F"iii\tEvaluations by depth: {moveInfo['heval_d']}")
            if(screenOut): print(trace[-1])
            trace.append(F"iv\tAverage evaluation depth: {moveInfo['d_avg']}")
            if(screenOut): print(trace[-1])
            trace.append(F"v\tAverage recursion depth: {moveInfo['r_d']}")
            if(screenOut): print(trace[-1])

            # if the time taken is > the specified time (and the player is not a human), the player loses
            if(end-start > self.t and not agents[self.player_turn].move == agents[self.player_turn].human):
                trace.append(F"agent {self.player_turn} loses: took too long")
                if(screenOut): print(trace[-1])
                return

            self.switch_player()



def main(n=4, b=(), s=4, d1=4, d2=4, t=5, a1=False, a2=True, mode=('H','H')):
    #initialize game
    g = Game(n,s,b,t)
    #play a game with two human players
    # g.play()
    #select the correct players

    agent_x = Agent('X', depth=d1, time=t)
    if(mode[0] == 'H'):
        agent_x.move = agent_x.human
    elif a1:
        agent_x.move = agent_x.heuristicless_alphabeta
    else:
        agent_x.move = agent_x.heuristicless_minimax


    agent_o = Agent('O', depth=d2, time=t)
    if(mode[1] == 'H'):
        agent_o.move = agent_o.human
    elif a2:
        agent_o.move = agent_o.heuristicless_alphabeta
    else:
        agent_o.move = agent_o.heuristicless_minimax

        
    g.play(agent_x=agent_x,agent_o=agent_o, log=True)

if __name__ == "__main__":
    #gather arguments
    args = {}
    args['n'] = int(sys.argv[1])
    #get list of blocs (I'm not actually sure the format of the blocs, but this is my best guess)
    b = int(sys.argv[2])
    args['b'] = []
    for i in range(b):
        args['b'].append(sys.argv[3+i].split(','))

    args['s'] = int(sys.argv[b+3])
    args['d1'] = int(sys.argv[b+4])
    args['d2'] = int(sys.argv[b+5])
    args['t'] = int(sys.argv[b+6])
    args['a1'] = sys.argv[b+7].lower() == 'true'
    args['a2'] = sys.argv[b+8].lower() == 'true'
    args['mode'] = sys.argv[b+9].split('-')

    #call main with the command line arguments
    main(**args)

