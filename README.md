# Gwent IA

Este repositório contém o desenvolvimento de agentes de **Deep Reinforcement Learning (DRL)** para o **GwentLite**, uma versão customizada e simplificada do jogo de cartas Gwent. 

O projeto foca em treinar redes neurais capazes de gerenciar vantagem de cartas, decidir o momento ideal de passar a rodada e utilizar efeitos especiais estrategicamente em uma disputa de melhor de três.

---

## 🚀 Treinamento em Supercomputador

Os modelos presentes neste repositório foram treinados em um ambiente de computação de alto desempenho (**Supercomputador/Cluster Slurm**), permitindo a exploração de arquiteturas complexas e grandes volumes de experiências (Experience Replay).

**Destaques do ambiente:**
- **Hardware:** GPUs de alto desempenho.
- **Software:** TensorFlow acelerado com CUDA 12.6 e XLA.
- **Escalabilidade:** Uso de gerenciador de tarefas Slurm para treinamentos de longa duração (10.000+ episódios).

---

## 🃏 GwentLite: Regras do Jogo

O **GwentLite** simplifica a mecânica do Gwent original para focar na lógica de decisão:

- **Cartas:** Representadas por números que indicam seu poder.
- **Cartas Especiais:**
    - **[3] Muster:** Joga todas as outras cópias de "3" do seu deck automaticamente.
    - **[6] Spy:** Dá 6 pontos ao oponente, mas permite que você compre 1 carta extra.
    - **[9] Scorch:** Adiciona 9 pontos ao seu lado e remove até 5 pontos do oponente.
- **Cartas Comuns:** Apenas adicionam seu valor nominal ao placar.
- **Objetivo:** Vencer 2 de 3 rodadas acumulando mais pontos que o adversário.

---

## 🧠 Agentes Disponíveis

- **DQN (Deep Q-Network):** Agente base com rede neural profunda.
- **DDQN (Double DQN):** Melhora a estabilidade ao evitar a superestimação de valores Q.
- **Dueling DQN:** Arquitetura que separa o valor do estado da vantagem da ação, ideal para jogos com estados de valor similar.
- **Minimax:** Um baseline clássico que utiliza busca em árvore com profundidade limitada para decisões táticas.

---

## 📂 Estrutura de Arquivos

- `agents/`: Implementações das arquiteturas de IA.
- `games/`: O motor do jogo `GwentLite.py`.
- `models/`: Pesos das redes neurais treinadas (ex: `DDQN_v2_10000.weights.h5`).
- `metrics/`: Logs de performance, ELO e resultados de torneios.
- `training_scripts/`: Scripts usados para treinar os agentes no cluster.
- `jogar_vs_ia.py`: Interface para desafiar um dos modelos treinados.

---

## 💾 Modelos Treinados

O repositório inclui modelos prontos para uso na pasta `models/`, treinados por 10.000 episódios cada:

- **`DQN_v1_10000.weights.h5`**: Agente DQN base treinado com recompensas padrão.
- **`DDQN_v2_10000.weights.h5`**: Agente Double DQN treinado com *Reward Shaping* para decisões mais agressivas e eficientes.

---

## 🛠️ Como Utilizar

### Requisitos
- Python 3.12+
- TensorFlow 2.16+
- NumPy

### Jogar contra a IA
Para testar suas estratégias contra o modelo treinado em supercomputador:
```bash
python jogar_vs_ia.py
```

### Executar Torneios
Para colocar os diferentes agentes para se enfrentarem:
```bash
python training_scripts/run_tournament_v2.py
```
