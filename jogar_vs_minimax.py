import sys
import time
sys.path.insert(0, 'games')
sys.path.insert(0, 'agents')

from GwentLite import GwentLite
from minimax_agent import MinimaxAgent

def print_game_state(env, player_id):
    """Imprime o estado do jogo de forma amigável para o humano."""
    print("\n" + "="*40)
    print(f"--- RODADA {env.round} ---")
    print("="*40)
    
    opponent_id = (player_id + 1) % 2
    
    print(f"\n[OPONENTE (Minimax)]")
    print(f"Pontos na Rodada: {env.player_points[opponent_id]}")
    print(f"Rodadas Vencidas: {env.player_num_round_wins[opponent_id]}")
    print(f"Cartas na Mão: {len(env.player_hands[opponent_id])}")
    print(f"Poder Restante (Deck): {env.player_total_remaining_card_power[opponent_id]}")
    
    print("-" * 20)
    
    print(f"\n[VOCÊ (Player {player_id})]")
    print(f"Pontos na Rodada: {env.player_points[player_id]}")
    print(f"Rodadas Vencidas: {env.player_num_round_wins[player_id]}")
    print(f"Sua Mão: {env.player_hands[player_id]}")
    print("-" * 20)
    
    # Mostra opções de jogada
    print("\nSUAS OPÇÕES:")
    print("0: PASSAR A VEZ")
    for i, card in enumerate(env.player_hands[player_id]):
        special = ""
        if card in env.special_cards:
            special = f"({env.special_cards[card]})"
        print(f"{i + 1}: Jogar carta {card} {special}")
    print("="*40)

def main():
    print("--- BEM-VINDO AO GWENT LITE VS MINIMAX ---")
    
    try:
        depth = int(input("Escolha a profundidade do Minimax (padrão 3, cuidado com >4): ") or 3)
    except:
        depth = 3
        
    print(f"Inicializando Minimax com profundidade {depth}...")
    agent = MinimaxAgent(depth=depth)
    env = GwentLite()
    
    # Escolha de ordem
    choice = input("Você quer ser o Player 0 (começa) ou Player 1? (0/1): ")
    human_player = 0 if choice == '0' else 1
    minimax_player = 1 if human_player == 0 else 0
    
    env.reset()
    
    game_over = False
    while not game_over:
        current_turn = env.get_player_turn()
        
        # Verifica se o jogo acabou antes de pedir ação
        game_over, results = env.check_game_over()
        if game_over:
            break

        if current_turn == human_player:
            print_game_state(env, human_player)
            
            valid_move = False
            while not valid_move:
                try:
                    move_str = input(f"Sua vez (0-{len(env.player_hands[human_player])}): ")
                    move = int(move_str)
                    
                    if env.act(move):
                        valid_move = True
                        print(f"\n>>> Você jogou: {'PASSAR' if move == 0 else env.player_hands[human_player][move-1] if move-1 < len(env.player_hands[human_player]) else 'carta'}")
                    else:
                        print("Jogada inválida (retorno interno False). Tente novamente.")
                        
                except (ValueError, IndexError):
                    print("Entrada inválida! Digite o número correspondente à ação.")
                except Exception as e:
                    print(f"Erro ao processar jogada: {e}")
                    pass
        else:
            print(f"\n[Minimax] Pensando... (Depth {depth})")
            start_time = time.time()
            action = agent.act(env)
            end_time = time.time()
            
            action_desc = "PASSAR"
            if action > 0:
                hand = env.player_hands[minimax_player]
                if action - 1 < len(hand):
                    card = hand[action - 1]
                    special = env.special_cards.get(card, "")
                    action_desc = f"Carta {card} {special}"
            
            print(f"[Minimax] Decidiu: {action_desc} (em {end_time - start_time:.2f}s)")
            
            env.act(action)
            
    print("\n" + "="*40)
    print("FIM DE JOGO!")
    print(f"Resultado: {results}")
    
    if results[human_player] == 'win':
        print("PARABÉNS! VOCÊ VENCEU O MINIMAX!")
    elif results[human_player] == 'tie':
        print("EMPATE!")
    else:
        print("VOCÊ PERDEU. MAIS SORTE NA PRÓXIMA!")
    print("="*40)

if __name__ == "__main__":
    main()
