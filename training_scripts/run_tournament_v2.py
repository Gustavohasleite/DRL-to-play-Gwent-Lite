import sys
import numpy as np
import os
import csv
import time
import itertools
from collections import defaultdict

sys.path.insert(0, 'games')
from GwentLite import GwentLite
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from minimax_agent import MinimaxAgent
from dueling_agent import DuelingAgent

# configurações
GAMES_PER_MATCHUP = 50
INITIAL_ELO = 1000
K_FACTOR = 32
MAX_TURNS = 100
OUTPUT_FILE = 'tournament_results_v2_depth3_run2.csv' # arquivo csv novo

# definição dos modelos
MODELS_CONFIG = {
    'DQN_v2_Fixed': { 'type': 'DQN', 'path': 'models_pro_DQN_v2_fixed/DQN_v2_10000.weights.h5' },
    'DDQN_v2_Fixed': { 'type': 'DDQN', 'path': 'models_pro_DDQN_v2_fixed/DDQN_v2_10000.weights.h5' },
    'DQN_v1_Pro': { 'type': 'DQN', 'path': 'models_pro_DQN_v1/DQN_v1_10000.weights.h5' },
    'DDQN_v1_Pro': { 'type': 'DDQN', 'path': 'models_pro_DDQN_v1/DDQN_v1_10000.weights.h5' },
    'Minimax_Depth3': { 'type': 'Minimax', 'path': None, 'depth': 3 }
}

def calculate_elo(p1_elo, p2_elo, p1_score):
    expected_p1 = 1 / (1 + 10**((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10**((p1_elo - p2_elo) / 400))
    new_p1_elo = p1_elo + K_FACTOR * (p1_score - expected_p1)
    new_p2_elo = p2_elo + K_FACTOR * ((1 - p1_score) - expected_p2)
    return new_p1_elo, new_p2_elo

def load_agent(name, config, state_size, action_size):
    print(f'Carregando agente {name}...')
    if config['type'] == 'Minimax': return MinimaxAgent(depth=config['depth'])

    if 'Fixed' in name or 'Pro' in name:
        is_double = (config['type'] == 'DDQN')
        agent = DuelingAgent(state_size, action_size, double_dqn=is_double)
    else:
        hidden_size = 256
        if config['type'] == 'DQN': agent = DQNAgent(state_size, action_size, hidden_size=hidden_size)
        else: agent = DDQNAgent(state_size, action_size, hidden_size=hidden_size)

    try:
        agent.load(config['path'])
        agent.epsilon = 0.0
    except Exception as e:
        print(f'ERRO CRÍTICO ao carregar {name}: {e}')
        return None
    return agent

def get_agent_action(agent, config, env, state):
    if config['type'] == 'Minimax': return agent.act(env)
    return agent.act(state)

def run_tournament():
    env = GwentLite()
    state_size = env.get_observation_shape()
    action_size = env.get_action_space_size()

    agents = {}
    valid_configs = {}
    for name, config in MODELS_CONFIG.items():
        agent = load_agent(name, config, state_size, action_size)
        if agent:
            agents[name] = agent
            valid_configs[name] = config

    if len(agents) < 2: return

    elos = {name: INITIAL_ELO for name in agents.keys()}
    matchups = list(itertools.permutations(agents.keys(), 2))

    print('-' * 60)
    print(f'INICIANDO TORNEIO ROUND-ROBIN (CORRIGIDO V2 - DEPTH 3 - RUN 2)')
    print('-' * 60)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Matchup_ID', 'Player0_Model', 'Player1_Model', 'Winner_Model', 'Turns',
                         'P0_ELO_Before', 'P1_ELO_Before', 'P0_ELO_After', 'P1_ELO_After', 'Termination'])

    for p0_name, p1_name in matchups:
        print(f'\n>>> Matchup: {p0_name} vs {p1_name}')
        agent0, config0 = agents[p0_name], valid_configs[p0_name]
        agent1, config1 = agents[p1_name], valid_configs[p1_name]
        wins_p0, wins_p1, ties = 0, 0, 0

        for i in range(GAMES_PER_MATCHUP):
            env.reset()
            done = False
            turns = 0
            winner_model = 'Tie'
            termination = 'Normal'

            while not done:
                if turns >= MAX_TURNS:
                    done = True; termination = 'Timeout'; break

                curr = env.get_player_turn()
                state = env.get_features(curr)
                action = get_agent_action(agent0, config0, env, state) if curr == 0 else get_agent_action(agent1, config1, env, state)

                if not env.act(action):
                    done = True; termination = 'Illegal_Move'
                    winner_model = p1_name if curr == 0 else p0_name
                else:
                    game_over, results = env.check_game_over()
                    if game_over:
                        done = True; termination = 'Game_Over'
                        if results[0] == 'win': winner_model = p0_name
                        elif results[1] == 'win': winner_model = p1_name
                        else: winner_model = 'Tie'
                turns += 1

            elo0, elo1 = elos[p0_name], elos[p1_name]
            score_p0 = 1.0 if winner_model == p0_name else (0.0 if winner_model == p1_name else 0.5)
            if score_p0 == 1.0: wins_p0 += 1
            elif score_p0 == 0.0: wins_p1 += 1
            else: ties += 1
            
            new_elo0, new_elo1 = calculate_elo(elo0, elo1, score_p0)
            elos[p0_name], elos[p1_name] = new_elo0, new_elo1

            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'{p0_name}_vs_{p1_name}', p0_name, p1_name, winner_model, turns,        
                                 f'{elo0:.2f}', f'{elo1:.2f}', f'{new_elo0:.2f}', f'{new_elo1:.2f}', termination])

    print('\n' + '='*50)
    print('CLASSIFICAÇÃO FINAL DO TORNEIO (ELO)')
    print('='*50)
    sorted_elos = sorted(elos.items(), key=lambda item: item[1], reverse=True)
    for rank, (name, elo) in enumerate(sorted_elos, 1):
        print(f'{rank}. {name}: {elo:.2f}')

if __name__ == '__main__':
    run_tournament()
