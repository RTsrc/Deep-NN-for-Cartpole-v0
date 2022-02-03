import matplotlib.pyplot as plt
import numpy as np
import gym
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
from keras.optimizers import Adam, Adagrad
from keras import initializers

epnum = 1000
stepnum = 500

class DQNet:
    def __init__(self):
        #set params
        self.HiddenNum = 10
        self.gamma = 0.99
        self.epsilon = 0.05
        self.learnRate = 0.001
        self.stateSize = 4
        self.actionNum = 2
        self.batchSize = 64
        self.bufferSize = 1024
        self._buffer = deque(maxlen=self.bufferSize)
        self.model = self.createModel()
        self.targetModel = self.createModel()
        self.UpdateFreq = 2
        
    #create a model
    def createModel(self):
        model = Sequential()
            
        model.add(Dense(self.HiddenNum, input_dim=self.stateSize, activation='relu')) 
        model.add(Dense(self.HiddenNum, activation='relu', kernel_initializer='random_uniform'))
        model.add(Dense(self.actionNum, activation='linear'))
        #compile the model
        model.compile(loss = 'mse', optimizer=Adam(lr=self.learnRate))
            
        return model
        
    def recall(self, state, action, reward, n_state, done):
        self._buffer.append((state, action, reward, n_state, done))
    
    def chooseAction(self, state,useTN=False):
        if random.random() < self.epsilon:
            return random.randrange(self.actionNum)
        else:
            if useTN:
                actions = self.targetModel.predict(state)
            else:
                actions = self.model.predict(state)
                
            action = np.argmax(actions[0])
            return action
    
    def UpdateQVals(self, state, reward, n_state, action, done, useTN=False):
        target = reward
        target_f = self.model.predict(state)
        
        if not useTN:
            QnMax = np.max(self.model.predict(n_state)[0])
        else:
            QnMax = np.max(self.targetModel.predict(n_state)[0])
            
            if not done:
                target = reward + self.gamma* QnMax
            
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        
    def Replay(self, bsize, useTN=False):
        if len(self._buffer) >= bsize:
            batch = random.sample(self._buffer, bsize)
            X_batch = []
            Y_batch = []
            for state, action, reward, n_state, done in batch:
                target = reward
                
                target_f = self.model.predict(state)
                
                if not useTN:
                    qVal = self.model.predict(n_state)[0]
                else:
                    qVal = self.targetModel.predict(n_state)[0]
                    
                if not done:
                    
                    QnMax = np.max(qVal)
                    target = reward + self.gamma * QnMax
                
                #Q(s,a) = r + gamma*max(Q(s', a'))
                
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
                
                #np.concatenate(X_batch, state, axis=0)
                #np.concatentate(Y_batch,target_f, axis=0)
                
            #self.model.fit(X_batch, Y_batch, batch_size=len(self._buffer), epochs=1, verbose=0)
            
    def backupNetwork(self, model, backup):
        weights = model.get_weights()
        backup.set_weights(weights)
        
    def updateTarget(self):
        self.backupNetwork(self.model, self.targetModel)
         
def simulate(epnum, stepnum, replay=False, target=False, updateFreq = 2):
    tLst = []
    rLst = []
    for i_episode in range(epnum):
        r_total = 0
        dr_total = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
                            
        for t in range(stepnum):
                                
            #env.render()
            action = agent.chooseAction(state, target)
                                
            nstate, reward, done, info = env.step(action)
            nstate = np.reshape(nstate, [1, 4])
                                
            if replay:
                agent.recall(state, action, reward, nstate, done)
            
            r_total += reward                        
            dr_total += agent.gamma ** (t+1) * reward
                
            if done:
                print("Total Reward=", r_total)
                print("Episode finished after {} timesteps".format(t+1))
                break
            
            if replay:
                agent.Replay(50, target)  
            else:
                agent.UpdateQVals(state,reward,nstate,action,done,target)
            
            #set state to the next state
            state = nstate
        
        #update the target net weights
        
        if target and i_episode%updateFreq == 0:
            print("Updating Target Net")
            agent.updateTarget()
        
        print("Episode:", i_episode, "Done")        
        print("Total Reward=", r_total) 
        print("Discounted Reward=", dr_total) 
        rLst.append(dr_total)
        tLst.append(i_episode)
        
    return tLst, rLst
if __name__ == '__main__':
    #env
    env = gym.make('CartPole-v0')
    agent = DQNet()
    #resu = simulate(1000,500)
    #resu = simulate(1000,500,True, False)
    #resu = simulate(1000,500,False, True)
    resu = simulate(2000,500,True, True)
    
    fig = plt.figure()
    plt.plot(resu[0], resu[1]) 
    #fig.suptitle('Cartpole V0 DQN: No Replay, No Target Network', fontsize=18)
    #fig.suptitle('Cartpole V0 DQN: Replay, No Target Network', fontsize=18)
    #fig.suptitle('Cartpole V0 DQN: No Replay, Target Network', fontsize=18)
    fig.suptitle('Cartpole V0 DQN: Replay, Target Network', fontsize=18)
    plt.xlabel('# of Episodes')
    plt.ylabel('Discounted Reward')
    #fig.savefig('graph1.jpg') 
    #fig.savefig('graph2.jpg') 
    #fig.savefig('graph3.jpg') 
    fig.savefig('graph4.jpg')    
    plt.show()