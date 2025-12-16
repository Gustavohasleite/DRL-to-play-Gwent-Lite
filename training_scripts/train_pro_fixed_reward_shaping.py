import sys
import numpy as np
import os
import csv
import time
import argparse
from collections import deque

# adiciona pasta games ao path
sys.path.insert(0, 'games')
from GwentLite import GwentLite
from dueling_agent import DuelingAgent

# configurações globais
EPISODES = 10000
TARGET_UPDATE_FREQ = 20
SAVE_MODEL_FREQ = 500
OPPONENT_UPDATE_FREQ = 500 # a cada 500 eps, o oponente vira a versão atual do agente
MAX_TURNS = 100
K_FACTOR = 32
INITIAL_ELO = 1000

def calculate_elo_update(p1_elo, p2_elo, p1_score, k_factor):
    expected_p1 = 1 / (1 + 10**((p2_elo - p1_elo) / 400))
    expected_p2 = 1 / (1 + 10**((p1_elo - p2_elo) / 400))
    new_p1_elo = p1_elo + k_factor * (p1_score - expected_p1)
    new_p2_elo = p2_elo + k_factor * ((1 - p1_score) - expected_p2)
    return new_p1_elo, new_p2_elo

def run_training(algorithm, reward_shaping):
    # setup de pastas e nomes
    is_double = (algorithm == 'DDQN')
    suffix = 'v2' if reward_shaping else 'v1'
    model_name = f"{algorithm}_{suffix}"
    save_dir = f"models_pro_{model_name}_fixed" # pasta nova para não misturar
    metrics_file = f"metrics_pro_{model_name}_fixed.csv"
    opponent_weights_file = f"temp_opponent_{model_name}_fixed.weights.h5"

    if not os.path.exists(save_dir): os.makedirs(save_dir)

    print(f"--- INICIANDO TREINO PRO V4 (FIXED): {model_name} ---")
    print(f"Algoritmo: {algorithm} (Dueling), Reward Shaping: {reward_shaping}")
    print(f"Episódios: {EPISODES}, Batch Size: 128")

    env = GwentLite()
    state_size = env.get_observation_shape()
    action_size = env.get_action_space_size()

    # agente principal
    agent = DuelingAgent(state_size, action_size, double_dqn=is_double)

    # agente oponente (começa como cópia do principal)
    opponent = DuelingAgent(state_size, action_size, double_dqn=is_double)
    opponent.epsilon = 0.1 # oponente joga bem, mas não perfeito

    # inicializa pesos do oponente
    agent.save(opponent_weights_file)
    opponent.load(opponent_weights_file)

    player_elos = {0: INITIAL_ELO, 1: INITIAL_ELO}

    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Epsilon', 'Winner', 'Turns', 'Total_Reward_P0', 'Duration_Sec', 'ELO_P0', 'ELO_P1'])

    start_time = time.time()

    for e in range(1, EPISODES + 1):
        env.reset()
        done = False
        turns = 0
        total_reward_p0 = 0

        # estado atual para fechar transições
        last_state_action = {0: None, 1: None} # p0 = agente, p1 = oponente
        
        # [fix] rastrear vitórias conhecidas
        last_known_wins = 0

        while not done:
            if turns >= MAX_TURNS:
                done = True
                winner = 'Timeout'
                break

            current_player = env.get_player_turn()
            state = env.get_features(current_player)

            # [fix] verificar se ganhamos um round
            # isso captura o caso onde passamos, o oponente jogou, o round acabou e nós ganhamos.
            if current_player == 0:
                current_wins = env.player_num_round_wins[0]
                if reward_shaping and current_wins > last_known_wins:
                    # ganhamos um round enquanto esperávamos!
                    # adiciona recompensa retroativa à ação anterior (o passe ou última carta)
                    if last_state_action[0] is not None:
                        s_prev, a_prev, r_prev = last_state_action[0]
                        # atualiza a recompensa da transição anterior
                        last_state_action[0] = (s_prev, a_prev, r_prev + 3.0)
                        total_reward_p0 += 3.0
                
                # atualiza nosso conhecimento
                last_known_wins = current_wins

            # escolha de ação
            if current_player == 0:
                # agente aprendiz
                action = agent.act(state)
            else:
                # oponente (fixed history)
                action = opponent.act(state)

            # memória do passo anterior (apenas para o agente p0)
            # salvamos a transição anterior agora que sabemos o novo estado e se houve recompensa extra
            if current_player == 0 and last_state_action[0] is not None:
                prev_s, prev_a, prev_r = last_state_action[0]
                agent.remember(prev_s, prev_a, prev_r, state, False)

            legal = env.act(action)
            turns += 1

            # --- cálculo de recompensa ---
            step_reward = 0
            if not legal:
                step_reward = -10
                done = True
                winner = 'Illegal'

                # punir e encerrar para p0
                if current_player == 0:
                    agent.remember(state, action, step_reward, state, True)
                    total_reward_p0 += step_reward
            else:
                # recompensa base
                step_reward = 0.1 # pequeno incentivo por jogar legal

                # [fix] removida a lógica antiga de reward shaping aqui
                # a verificação wins_after > wins_before era falha para passes.
                # agora tratamos isso no início do turno (acima) ou no game over (abaixo).

                game_over, results = env.check_game_over()

                if game_over:
                    done = True

                    # [fix] verificar se ganhamos o último round no momento do game over
                    # (caso a vitória do jogo coincida com a vitória do round)
                    if current_player == 0:
                        current_wins = env.player_num_round_wins[0]
                        if reward_shaping and current_wins > last_known_wins:
                            step_reward += 3.0 # adiciona ao step atual pois não haverá próximo
                            total_reward_p0 += 3.0

                    # recompensa final
                    final_r = 5.0 if results[current_player] == 'win' else (-5.0 if results[current_player] == 'loss' else 0)
                    total_r = step_reward + final_r

                    if current_player == 0:
                        agent.remember(state, action, total_r, state, True)
                        total_reward_p0 += total_r

                    winner = 0 if results[0] == 'win' else (1 if results[1] == 'win' else 'Tie')

                    if winner != 'Illegal':
                        p0_score = 1 if winner == 0 else (0 if winner == 1 else 0.5)
                        player_elos[0], player_elos[1] = calculate_elo_update(player_elos[0], player_elos[1], p0_score, K_FACTOR)

                else:
                    # jogo segue
                    if current_player == 0:
                        # guardamos este estado/ação para fechar a transição no próximo turno
                        # (ou adicionar recompensa de round se acontecer nesse intervalo)
                        last_state_action[0] = (state, action, step_reward)
                        total_reward_p0 += step_reward

            # treinar
            if current_player == 0 or done:
                agent.replay()

        # --- fim do episódio ---

        # atualizar target network
        if e % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        # atualizar oponente (curriculum)
        if e % OPPONENT_UPDATE_FREQ == 0:
            print(f">> Atualizando Oponente com versão do Ep {e}")
            agent.save(opponent_weights_file)
            opponent.load(opponent_weights_file)

        # decaimento epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # logs
        duration = time.time() - start_time
        if e % 50 == 0:
            print(f"Ep {e}/{EPISODES} | {model_name} | Win: {winner} | Ep: {agent.epsilon:.2f} | R: {total_reward_p0:.1f} | ELO P0: {player_elos[0]:.0f}")

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e, agent.epsilon, winner, turns, total_reward_p0, duration, player_elos[0], player_elos[1]])

        # checkpoints
        if e % SAVE_MODEL_FREQ == 0:
            agent.save(f"{save_dir}/{model_name}_{e}.weights.h5")

    print(f"Treinamento {model_name} Concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['DQN', 'DDQN'], required=True)
    parser.add_argument("--shaping", type=str, choices=['True', 'False'], required=True)
    args = parser.parse_args()

    use_shaping = (args.shaping == 'True')
    run_training(args.type, use_shaping)
