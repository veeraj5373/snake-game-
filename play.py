import torch
import random
import numpy as np
from collections import deque
from snakegame import SnakeGame_AI,Direction,Point
from model import Lineat_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
        self.n_games =0
        self.epsilon=0 #randomness
        self.gamma =0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY)# popleft()
        self.model=Lineat_QNet(11,256,3)
        self.model.load_state_dict(torch.load('../game/model/best_model.pth'))

        self.model.eval()
        self.trainer = QTrainer(self.model, LR, self.gamma)


    def get_state(self,snakegame):
        head= snakegame.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = snakegame.direction == Direction.LEFT
        dir_r = snakegame.direction == Direction.RIGHT
        dir_u = snakegame.direction == Direction.UP
        dir_d = snakegame.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and snakegame.is_collision(point_r)) or
            (dir_l and snakegame.is_collision(point_l)) or
            (dir_u and snakegame.is_collision(point_u)) or
            (dir_d and snakegame.is_collision(point_d)),

            # Danger right
            (dir_u and snakegame.is_collision(point_r)) or
            (dir_d and snakegame.is_collision(point_l)) or
            (dir_l and snakegame.is_collision(point_u)) or
            (dir_r and snakegame.is_collision(point_d)),

            # Danger left
            (dir_d and snakegame.is_collision(point_r)) or
            (dir_u and snakegame.is_collision(point_l)) or
            (dir_r and snakegame.is_collision(point_u)) or
            (dir_l and snakegame.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            snakegame.food.x < snakegame.head.x,  # food left
            snakegame.food.x > snakegame.head.x,  # food right
            snakegame.food.y < snakegame.head.y,  # food up
            snakegame.food.y > snakegame.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))#popleft if MAXMEMORY reached
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory , BATCH_SIZE)#list of tuple
        else:
            mini_sample=self.memory

        states, actions,rewards, next_states, dones=zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state, action, reward, next_state, done)
        pass


    def get_action(self,state):
        self.epsilon= 80-self.n_games
        final_move =[0,0,0]
        if random.randint(0,200)< self.epsilon:
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move= torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train():
    plot_scores=[]
    plot_mean_score=[]
    total_score=0
    record=0
    agent=Agent()
    game= SnakeGame_AI()
    while True:
        #get old state
        state_old = agent.get_state(game)
        # get move
        final_move =agent.get_action(state_old)

        #performa move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        agent.remember(state_old,final_move,reward,state_new,done)

        if done:

            #train the long memory
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()

            if score> reward:
                record = score
                #agent.model.save()
                agent.model.save("best_model.pth")
            print('Game',agent.n_games,"Score",score,'Record:',record)

            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores,plot_mean_score)



if __name__=="__main__":
    train()