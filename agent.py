import torch
import random
import numpy as np
from collections import deque
from game import Connect4_AI_Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #num_states = [6,7]
        self.model = Linear_QNet(42, 50, 50, 7)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def agent_move(self, game, predictedmove):
        game.move(predictedmove)

    def get_state(self, game):
        return game.get_board_state()

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 6)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)



def train():
    total_wins = 0
    ai_wins = 0
    reward = 0
    agent = Agent()
    game = Connect4_AI_Game()

    while True:

        turn = game.turn

        if turn == game.AI and not game.game_over:
            reward, done, winner = game.ai_step(reward)
            print("CURRENT REWARD:")
            print(reward)


        elif turn == game.PLAYER and not game.game_over:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)
            reward, done, winner = game.play_step(final_move, reward)

            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
            print("CURRENT REWARD:")
            print(reward)





        elif done:
            game.check_game_over()
            # train long memory, plot result
            #game.reset()
            agent.train_long_memory()
            agent.n_games += 1
            #agent.train_long_memory()
            if winner == "PLAYER":
                total_wins += 1
            if winner == "AI":
                ai_wins += 1
            print('Game', agent.n_games, 'Wins:', total_wins)
            reward = 0
            game.reset()







if __name__ == '__main__':
    train()
