"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return number_of_free_adjacent_cells(game, player)

def number_of_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function calculates the difference between the number of moves left for each player and returns the result. It's a simple heuristic that based on the intuition that the player with the most number of legal moves left is more likely to win.

	
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")
    
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
		
    if player_moves == 0:
        return float("-inf")
		
    heuristic_value = float(player_moves - opponent_moves)
		
    return heuristic_value

def number_of_moves_with_bias(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function calculates the difference between the number of moves left for each player and returns the result with a bias for the player moves. It's a variation of the number_moves_heuristic which is designed to favor the player.

	
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")
    
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
		
    if player_moves == 0:
        return float("-inf")
		
    heuristic_value = float(3/4*player_moves - 1/4*opponent_moves)
		
    return heuristic_value
	
def number_of_unique_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function calculates the difference between the number of moves left for each player and returns the result, removing the moves they have in common. It's a version of the mumber_of_moves heuristic that favors unique moves.

	
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")
   
    common_moves = set(game.get_legal_moves(player)).intersection(game.get_legal_moves(game.get_opponent(player)))

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
		
    if player_moves == 0:
        return float("-inf")
    
    unique_player_moves = player_moves - len(common_moves)
    unique_opponent_moves = opponent_moves - len(common_moves)
		
    heuristic_value = float(unique_player_moves - unique_opponent_moves)
		
    return heuristic_value

def number_of_free_adjacent_cells(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function calculates the difference between the number of moves left for each player and returns the result, adding the number of free adjacent moves to this number and then substracting the resulting values.

	
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")
   
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
		
    if player_moves == 0:
        return float("-inf")
    
    player_value = 0
    opponent_value = 0
	
    for x_position,y_position in player_moves:
        player_box_positions = [(x_position+i,y_position+j) for i in range(-1,2) for j in range(-1,2)] 
        if len(player_box_positions) == 8:
            player_value = player_value + 1
			
    for x_position,y_position in opponent_moves:
        opponent_box_positions = [(x_position+i,y_position+j) for i in range(-1,2) for j in range(-1,2)] 
        if len(opponent_box_positions) == 8:
            opponent_value = opponent_value + 1
		
    heuristic_value = float(len(player_moves)+player_value - len(opponent_moves)-opponent_value)
		
    return heuristic_value
	
class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
		
		#Return immediately if there are no legal moves left
        if not legal_moves:
            return(-1, -1)
			
		#Initial value of the move
        move = (-1,-1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
			
			#If self.iterative=True perform iterative deepening to a maximum number of plies
            if self.iterative:
                if self.method == 'minimax':
                    for iterative_depth in range(1,8):
                        score,move = self.minimax(game,iterative_depth)
                if self.method == 'alphabeta':
                    for iterative_depth in range(1,8):
                        score,move = self.alphabeta(game,iterative_depth)
            #If self.iterative=False do not perform iterative deepening
            else:
                if self.method == 'minimax':
                    score,move = self.minimax(game,self.search_depth)
                else:
                    if self.method == 'alphabeta':
                        score,move = self.alphabeta(game,self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #The algorith used is taken in part from http://aima.cs.berkeley.edu/python/games.html 
        #Terminal test value
        legal_moves = game.get_legal_moves()
        #Game utility when terminal in this case no legal moves are left
        game_utility_terminal = game.utility(self), (-1,-1)


        #Test to see if arrived at the end of the search in the game tree
        if depth == 0:
		
        #End of game, return score and best move for current location of active player
            return self.score(game, self), game.get_player_location(self)


        #If the maximum number of plies has not been reach then decide on next move 
        #for maximizing or minimizing layer
        #If search depth at a maximizing layer then return max value
        if maximizing_player:
		
            #Starting score is "-inf"
            new_score =  float("-inf")
            new_move = (-1,-1)
			
            #For each legal move left 
            for possible_move in legal_moves:
			
            #Get the new board with the possible_move applied
                new_board = game.forecast_move(possible_move)
				
                #Get score for new board for minimizing layer
                search_score, search_move = self.minimax(new_board,depth-1,False)
				
                #Get highest score value from possible moves or "-inf", and corresponding move
                if search_score > new_score:
                    new_score,new_move = search_score,possible_move

        
        #If search depth at a minimizing layer then return max value
        else:
            #Starting score is "inf"
            new_score =  float("inf")
            new_move = (-1,-1)
			
            #For each legal move left 
            for possible_move in legal_moves:
			
                #Get the new board with the possible_move applied
                new_board = game.forecast_move(possible_move)
				
                #Get score for new board for maximizing layer
                search_score, search_move = self.minimax(new_board, depth - 1 , True)
				
                #Get lowest score value from possible moves or "inf", and corresponding move
                if search_score < new_score:
                    new_score, new_move = search_score, possible_move

        #return score,best move for current location of active player
        return new_score, new_move
		    
		    
		

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #The algorith used is taken in part from http://aima.cs.berkeley.edu/python/games.html 
        #Terminal test value
        legal_moves = game.get_legal_moves()
        #Game utility when terminal in this case no legal moves are left
        game_utility_terminal = game.utility(self), (-1,-1)


        #Test to see if arrived at the end of the search in the game tree
        if depth == 0:
        #End of game, return score and best move for current location of active player
            return self.score(game, self), game.get_player_location(self)


        #If the maximum number of plies has not been reach then decide on next move 
        #for maximizing or minimizing layer
        #If search depth at a maximizing layer then return max value
        if maximizing_player:
		
            #Starting score is "-inf"
            new_score =  float("-inf")
            new_move = (-1,-1)
			
            #For each legal move left 
            for possible_move in legal_moves:
			
            #Get the new board with the possible_move applied
                new_board = game.forecast_move(possible_move)
				
                #Get score for new board for minimizing layer
                search_score, search_move = self.alphabeta(new_board,depth-1, alpha, beta, False)
				
                #Get highest score value from possible moves or "-inf", and corresponding move
                if search_score > new_score:
                    new_score,new_move = search_score,possible_move
					
				#Add alphabeta pruning to mininmax
                if new_score >= beta:
                    alpha = max(alpha, new_score)
				
				

        
        #If search depth at a minimizing layer then return max value
        else:
            #Starting score is "inf"
            new_score =  float("inf")
            new_move = (-1,-1)
			
            #For each legal move left 
            for possible_move in legal_moves:
			
                #Get the new board with the possible_move applied
                new_board = game.forecast_move(possible_move)
				
                #Get score for new board for maximizing layer
                search_score, search_move = self.alphabeta(new_board, alpha, beta, depth - 1 , True)
				
                #Get lowest score value from possible moves or "inf", and corresponding move
                if search_score < new_score:
                    new_score, new_move = search_score, possible_move
					
			    #Add alphabeta pruning to mininmax
                if new_score <= alpha:
                    beta = min(beta, new_score)

        #return score,best move for current location of active player
        return new_score, new_move