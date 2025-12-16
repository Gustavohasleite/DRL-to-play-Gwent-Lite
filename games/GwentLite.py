from games.Game import *
import numpy as np

class Deck:
    def __init__(self,min_deck_size,max_deck_power):
        self.min_deck_size = min_deck_size
        self.max_deck_power = max_deck_power
        self.max_card_power = self.max_deck_power - self.min_deck_size + 1
        
        self.deck = []
        self.deck_size = 0
        self.feature_representation = np.array([0 for _ in range(self.max_card_power)],dtype=float)
    def reset(self,input_deck_list,mean=None,stdev=None):
        
        if input_deck_list != None: deck_list = input_deck_list.copy()
        else:
            deck_list = []
            max_card_power = self.max_card_power
            while len(deck_list) < self.min_deck_size:
                if max_card_power == 1: deck_list.append(1)
                else:
                    card = max( 1 , round( np.random.normal(mean,stdev) ) )
                    card = min( max_card_power , card )
                    deck_list.append(card)
                    max_card_power -= card - 1
            deck_list[-1] += max_card_power - 1

        self.deck = deck_list
        self.deck_size = len(deck_list)
        self.feature_representation *= 0
        for card in self.deck: self.feature_representation[card-1] += 1
        np.random.shuffle(self.deck) 
    def draw(self):
        card = self.deck.pop(-1)
        self.deck_size -= 1
        self.feature_representation[card-1] -= 1
        return card
    def get_features(self):
        if self.deck_size == 0: return self.feature_representation * 0 
        return self.feature_representation / self.deck_size
    def __str__(self): return str(self.deck)
    def __repr__(self): return str(self)

class GwentLite(Game):
    def __init__(self):
        self.min_deck_size = 25
        self.max_deck_power = 100
        self.max_card_power = self.max_deck_power - self.min_deck_size + 1
        self.mean = round(self.max_deck_power/self.min_deck_size)
        self.stdev_range = range( 0 , round(self.max_deck_power/self.min_deck_size*16/3/3)+1 )

        self.player_decks = { 0 : Deck(self.min_deck_size,self.max_deck_power) , 1 : Deck(self.min_deck_size,self.max_deck_power) }
        self.num_unplayed_cards = { 0 : 0 , 1 : 0 }
        self.player_hands = { 0 : [] , 1 : [] }
        self.player_points = { 0 : 0 , 1 : 0 }
        self.player_num_round_wins = { 0 : 0 , 1 : 0 }
        self.player_total_remaining_card_power = { 0 : 0 , 1 : 0 }
        self.player_average_remaining_card_power = { 0 : 0 , 1 : 0 }
        
        self.round = 0
        self.active_players = []
        self.active_player_index = None
        self.round_one_first_player_index = None
        
        self.scorch_damage = 5
        self.special_cards = {
            3: 'MUSTER',
            6: 'SPY',
            9: 'SCORCH'
        }
        
    def get_name(self): return 'Gwent Lite'
    def get_observation_shape(self): return (self.max_card_power+10+5) + (7) + (2)
    def get_action_space_size(self): return 11
    def get_number_of_players(self): return 2
    def reset(self,deck_lists=(None,None)):
        for player_index in range(2):
            self.player_decks[player_index].reset(deck_lists[player_index],self.mean,np.random.choice(self.stdev_range))
            self.num_unplayed_cards[player_index] = len(self.player_decks[player_index].deck)
            self.player_hands[player_index].clear()
            for _ in range(10): self.player_hands[player_index].append( self.player_decks[player_index].draw() )
            self.player_points[player_index] = 0
            self.player_num_round_wins[player_index] = 0
            self.player_total_remaining_card_power[player_index] = self.max_deck_power
            self.player_average_remaining_card_power[player_index] = self.player_total_remaining_card_power[player_index] / self.num_unplayed_cards[player_index]

        self.round = 1
        self.active_players = [0,1]
        self.active_player_index = np.random.choice(2)
        self.round_one_first_player_index = self.get_player_turn()
    def next_round(self):
        self.round += 1
        self.active_players = [0,1]

        if self.player_points[0] == self.player_points[1]:
            for player_index in range(2): self.player_num_round_wins[player_index] += 1
            self.active_player_index = (self.round_one_first_player_index+1) % 2
        else:
            winner_index = np.argmax( ( self.player_points[0] , self.player_points[1] ) )
            self.player_num_round_wins[winner_index] += 1
            self.active_player_index = winner_index

        for player_index in range(2):
            self.player_points[player_index] = 0
            for _ in range( min( 10-len(self.player_hands[player_index]) , 3 ) ): 
                if self.player_decks[player_index].deck_size > 0:
                    self.player_hands[player_index].append( self.player_decks[player_index].draw() )
    def get_player_turn(self): return self.active_players[self.active_player_index]
    def act(self,action):
        active_player_index = self.get_player_turn()

        if action == 0 or action-1 >= len(self.player_hands[ active_player_index ]):
            self.active_players.remove( active_player_index )
            if len(self.active_players) == 0: self.next_round()
            else: self.active_player_index = (self.active_player_index+1) % len(self.active_players)
            return True

        if action-1 < len(self.player_hands[ active_player_index ]):
            card_played = self.player_hands[ active_player_index ].pop(action-1)
            
            ability = self.special_cards.get(card_played)
            opponent_index = (active_player_index + 1) % 2
            
            if ability == 'SPY':
                self.player_points[ opponent_index ] += card_played
                if self.player_decks[active_player_index].deck_size > 0:
                     self.player_hands[active_player_index].append( self.player_decks[active_player_index].draw() )
            
            elif ability == 'SCORCH':
                self.player_points[ active_player_index ] += card_played
                damage = min(self.player_points[ opponent_index ], self.scorch_damage)
                self.player_points[ opponent_index ] -= damage

            elif ability == 'MUSTER':
                self.player_points[ active_player_index ] += card_played
                
                deck_ref = self.player_decks[active_player_index]
                indices_to_remove = [i for i, x in enumerate(deck_ref.deck) if x == card_played]
                
                for i in reversed(indices_to_remove):
                    val = deck_ref.deck.pop(i)
                    deck_ref.deck_size -= 1
                    deck_ref.feature_representation[val-1] -= 1
                    
                    self.player_points[ active_player_index ] += val
                    self.num_unplayed_cards[ active_player_index ] -= 1
                    self.player_total_remaining_card_power[ active_player_index ] -= val

            else:
                self.player_points[ active_player_index ] += card_played 
            
            self.num_unplayed_cards[ active_player_index ] -= 1
            self.player_total_remaining_card_power[ active_player_index ] -= card_played
            
            if self.num_unplayed_cards[ active_player_index ] > 0:
                self.player_average_remaining_card_power[ active_player_index ] = self.player_total_remaining_card_power[ active_player_index ] / self.num_unplayed_cards[ active_player_index ]
            else:
                self.player_average_remaining_card_power[ active_player_index ] = 0

            if len(self.player_hands[ active_player_index ]) == 0:
                self.active_players.remove( active_player_index )
                if len(self.active_players) == 0: self.next_round()
                else: self.active_player_index = (self.active_player_index+1) % len(self.active_players)
            else: self.active_player_index = (self.active_player_index+1) % len(self.active_players)
            
            return True

    def check_game_over(self):
        players_with_2_round_wins = []
        for player_index in range(2):
            if self.player_num_round_wins[player_index] == 2: players_with_2_round_wins.append(player_index)

        if len(players_with_2_round_wins) == 0: return False,None
        if len(players_with_2_round_wins) == 1: return True , { players_with_2_round_wins[0] : 'win' , (players_with_2_round_wins[0]+1)%2 : 'loss' }

        return True , {0:'tie',1:'tie'}
    def get_features(self,player_index):
        
        opponent_index = (player_index+1) % 2

        return np.concatenate( (
            
            self.player_decks[player_index].get_features(),
            [ self.player_hands[player_index][i]/self.max_card_power if i < len(self.player_hands[player_index]) else 0 for i in range(10) ],
            [ self.num_unplayed_cards[player_index] / self.min_deck_size ,
              self.player_points[player_index] / ( self.max_card_power + 9 ),
              self.player_num_round_wins[player_index] / 2,
              self.player_total_remaining_card_power[player_index] / self.max_deck_power,
              self.player_average_remaining_card_power[player_index] / self.max_card_power ],

            [ len(self.player_decks[opponent_index].deck) / self.min_deck_size,
              len(self.player_hands[opponent_index]) / 10,
              self.num_unplayed_cards[opponent_index] / self.min_deck_size ,
              self.player_points[opponent_index] / ( self.max_card_power + 9 ),
              self.player_num_round_wins[opponent_index] / 2,
              self.player_total_remaining_card_power[opponent_index] / self.max_deck_power,
              self.player_average_remaining_card_power[opponent_index] / self.max_card_power ],

            [ self.round / 3 , len(self.active_players) / 2 ]

            ) ).reshape(1,-1)

    def sample_legal_move(self):
        return np.random.choice( len( self.player_hands[ self.get_player_turn() ] ) + 1 )
    def __str__(self): return f'\nPlayer decks: {self.player_decks}\nPlayer hands: {self.player_hands}\nPlayer points: {self.player_points}\nPlayer round wins: {self.player_num_round_wins}\nPlayer total remaining card power: {self.player_total_remaining_card_power}\nPlayer average remaining card power: {self.player_average_remaining_card_power}\nRound: {self.round}\nPlayer turn: {self.get_player_turn()}'
    def __repr__(self): return str(self)
    def play(self):
        while True:
            self.reset()

            game_over,winner = self.check_game_over()
            while not game_over:
                print(self)
                self.get_features(self.get_player_turn())
                
                a = int( input('Move: ') )
                while not self.act(a): a = int( input('Illegal!\nMove: ') )
                game_over,winner = self.check_game_over()
            print(self)
            self.get_features(self.get_player_turn())

            print(self.check_game_over())

if __name__ == '__main__':
    g = GwentLite()
    g.play()