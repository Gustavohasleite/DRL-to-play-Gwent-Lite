import sys
import time
import os
import glob
import numpy as np

sys.path.insert(0, 'games')
sys.path.insert(0, 'agents')

from GwentLite import GwentLite
from minimax_agent import MinimaxAgent
from dueling_agent import DuelingAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def print_game_state(env, player_id):
    """Imprime o estado do jogo de forma amigável para o humano."""
    print("\n" + "="*50)
    print(f"--- RODADA {env.round} ---")
    print("="*50)
    
    opponent_id = (player_id + 1) % 2
    
    # OPONENTE
    print(f"\n>>> MESA DO OPONENTE (IA)")
    print(f"| PODER TOTAL NA MESA:  {env.player_points[opponent_id]:<3} |")
    print(f"---------------------------------")
    print(f"Rodadas Vencidas: {env.player_num_round_wins[opponent_id]}")
    print(f"Cartas na Mão:    {len(env.player_hands[opponent_id])}")
    print(f"Deck Restante:    {env.player_total_remaining_card_power[opponent_id]} pts")
    
    print("\n" + " "*20 + "VS" + " "*20)
    
    # JOGADOR
    print(f"\n>>> SUA MESA (Player {player_id})")
    print(f"| PODER TOTAL NA MESA:  {env.player_points[player_id]:<3} |")
    print(f"---------------------------------")
    print(f"Rodadas Vencidas: {env.player_num_round_wins[player_id]}")
    print(f"Sua Mão:          {env.player_hands[player_id]}")
    print("-" * 50)
    
    # OPÇÕES
    print("SUAS OPÇÕES DE JOGADA:")
    print(" 0: PASSAR A VEZ")
    for i, card in enumerate(env.player_hands[player_id]):
        special = ""
        if card in env.special_cards:
            special = f" ({env.special_cards[card]})"
        print(f"{i + 1:>2}: Jogar carta [{card}]{special}")
    print("="*50)

def get_available_models():
    """Busca arquivos de modelo .h5 na pasta atual e subpastas."""
    models = []
    search_path = os.path.join(os.getcwd(), "**", "*.weights.h5")
    for file in glob.glob(search_path, recursive=True):
        if "Gwent IA Definitivo" in file:
             models.append(file)
    return sorted(models, key=os.path.getmtime, reverse=True)

def main():
    print("--- GWENT LITE: HUMANO VS IA ---")
    
    print("\nEscolha seu oponente:")
    print("1. Minimax (Algoritmo de busca)")
    print("2. Carregar Modelo Treinado (DQN/DDQN)")
    
    opponent_type = input("Opção (1/2): ")
    
    agent = None
    agent_name = "IA"
    
    env = GwentLite() 
    
    if opponent_type == '1':
        try:
            depth = int(input("Profundidade do Minimax (Enter para 3): ") or 3)
        except:
            depth = 3
        agent = MinimaxAgent(depth=depth)
        agent_name = f"Minimax-d{depth}"
        
    elif opponent_type == '2':
        models = get_available_models()
        if not models:
            print("Nenhum modelo .weights.h5 encontrado!")
            return

        print("\nModelos encontrados:")
        for i, m in enumerate(models):
            print(f"{i+1}. {os.path.basename(m)}")
        
        try:
            sel = int(input("\nEscolha o número do modelo: ")) - 1
            if 0 <= sel < len(models):
                model_path = models[sel]
                filename = os.path.basename(model_path)
                is_double = 'DDQN' in filename or 'ddqn' in filename
                
                print(f"Carregando {filename}...")
                
                state_size = env.get_observation_shape()
                action_size = env.get_action_space_size()
                
                agent = DuelingAgent(state_size, action_size, double_dqn=is_double)
                agent.load(model_path)
                agent.epsilon = 0.0
                
                agent_name = f"IA-{filename}"
            else:
                print("Seleção inválida.")
                return
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return
    else:
        print("Opção inválida.")
        return

    choice = input("\nVocê quer ser o Player 0 (começa) ou Player 1? (0/1): ")
    human_player = 0 if choice == '0' else 1
    ai_player = 1 if human_player == 0 else 0
    
    print(f"\nIniciando jogo: Você (P{human_player}) vs {agent_name} (P{ai_player})")
    
    env.reset()
    current_round_tracker = env.round
    
    game_over = False
    while not game_over:
        current_turn = env.get_player_turn()
        
        game_over, results = env.check_game_over()
        if game_over:
            break
            
        if env.round > current_round_tracker:
            print("\n" + "*"*50)
            print(f"   FIM DA RODADA {current_round_tracker}!")
            print(f"   Placar de Rodadas: VOCÊ {env.player_num_round_wins[human_player]} x {env.player_num_round_wins[ai_player]} IA")
            print(f"   >>> Limpando a mesa e comprando cartas... <<<")
            print("*"*50 + "\n")
            time.sleep(2)
            current_round_tracker = env.round

        if current_turn == human_player:
            print_game_state(env, human_player)
            
            valid_move = False
            while not valid_move:
                try:
                    move_str = input(f"Sua vez (0 para Passar): ")
                    move = int(move_str)
                    
                    if env.act(move):
                        valid_move = True
                        played = "PASSOU A VEZ"
                        if move > 0:
                            # Recupera carta jogada (agora já foi removida da mão, mas printamos o índice antes)
                            # Para consistência visual:
                            pass # Apenas segue
                        print(f"\n>>> VOCÊ: Escolheu ação {move}")
                    else:
                        print("Jogada inválida! Tente novamente.")
                        
                except (ValueError, IndexError):
                    print("Entrada inválida!")
        else:
            print(f"\n[{agent_name}] Pensando...")
            time.sleep(1.5)
            
            if isinstance(agent, MinimaxAgent):
                action = agent.act(env)
            else:
                state = env.get_features(ai_player)
                action = agent.act(state)
            
            # Identificar a carta antes de jogar
            action_desc = "PASSOU A VEZ"
            if action > 0:
                hand = env.player_hands[ai_player]
                # Verifica se índice é válido
                if action - 1 < len(hand):
                    card_val = hand[action - 1]
                    special = env.special_cards.get(card_val, "")
                    effect_str = f" ({special})" if special else ""
                    action_desc = f"JOGOU A CARTA [{card_val}]{effect_str}"
                else:
                    action_desc = f"Tentou jogar índice {action} (Inválido)"

            print(f">>> {agent_name}: {action_desc}")
            env.act(action)
            
    print("\n" + "="*50)
    print("FIM DE JOGO!")
    print(f"Resultado: {results}")
    
    if results[human_player] == 'win':
        print("PARABÉNS! VOCÊ VENCEU A IA!")
    elif results[human_player] == 'tie':
        print("EMPATE!")
    else:
        print("VOCÊ PERDEU. A IA VENCEU!")
    print("="*50)

if __name__ == "__main__":
    main()
