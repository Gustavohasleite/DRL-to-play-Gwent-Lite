import copy
import math
import numpy as np

class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth
        self.expanded_nodes = 0

    def act(self, env):
        # calculates the best move for the current player using minimax with alpha-beta pruning
        current_player = env.get_player_turn()
        self.expanded_nodes = 0

        # get legal actions: 0 (pass) + indices for cards in hand (1..len(hand))
        hand_size = len(env.player_hands[current_player])
        legal_actions = [0] + list(range(1, hand_size + 1))

        best_score = -math.inf
        best_action = 0 # default to pass if everything else fails

        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            # clone the environment to simulate the move
            env_copy = copy.deepcopy(env)

            # apply move
            # note: act() returns true/false, but for minimax we assume we picked from legal_actions
            # gwentlite handles turn switching internally
            env_copy.act(action)

            # recursively call minimax
            # if the move ended the game or round, the env_copy state reflects that
            score = self.minimax(env_copy, self.depth - 1, alpha, beta, False, current_player)

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break # beta cutoff (pruning)

        return best_action

    def minimax(self, env, depth, alpha, beta, is_maximizing, player_id):
        self.expanded_nodes += 1

        game_over, winner = env.check_game_over()
        if depth == 0 or game_over:
            return self.evaluate(env, player_id)

        # who is the *current* active player in the simulation?
        # note: in gwent, one player might pass, leaving the other to play multiple turns.
        # so 'is_maximizing' depends on whether the simulation's current turn belongs to 'player_id'
        
        sim_current_player = env.get_player_turn()
        
        # if sim_current_player is the agent (player_id), we maximize. else minimize.
        is_current_maximizing = (sim_current_player == player_id)
        
        hand_size = len(env.player_hands[sim_current_player])
        legal_actions = [0] + list(range(1, hand_size + 1))
        
        if is_current_maximizing:
            max_eval = -math.inf
            for action in legal_actions:
                env_copy = copy.deepcopy(env)
                env_copy.act(action)
                eval_score = self.minimax(env_copy, depth - 1, alpha, beta, False, player_id)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for action in legal_actions:
                env_copy = copy.deepcopy(env)
                env_copy.act(action)
                eval_score = self.minimax(env_copy, depth - 1, alpha, beta, True, player_id)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, env, player_id):
        # heuristic evaluation function
        # positive value = good for player_id
        # negative value = good for opponent
        opponent_id = (player_id + 1) % 2
        
        # 1. game over check
        game_over, winner = env.check_game_over()
        if game_over:
            if winner[player_id] == 'win':
                return 100000 # massive score for winning game
            elif winner[player_id] == 'loss':
                return -100000
            else:
                return 0 # tie

        score = 0
        
        # 2. round wins (most important)
        score += env.player_num_round_wins[player_id] * 10000
        score -= env.player_num_round_wins[opponent_id] * 10000
        
        # 3. current round points
        # only relevant if the round is still going
        score += env.player_points[player_id] * 10
        score -= env.player_points[opponent_id] * 10
        
        # 4. card advantage (critical in gwent)
        # more cards = more flexibility later
        score += len(env.player_hands[player_id]) * 50
        score -= len(env.player_hands[opponent_id]) * 50
        
        return score