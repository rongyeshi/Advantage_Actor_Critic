import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from reinforce import Reinforce
from keras.optimizers import Adam
import keras.backend as K

from keras.layers import Input
#from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Model

input_size  = 8
output_size = 4

gamma = 0.99
num_epochs =5
num_epochs2 = 5
k=100
lr_set =1e-4
c_lr_set = 5e-4
n_set=100
low_bound = 1e-15#5



def cust_loss_arctor(y_true, y_pred):
    
    GG = y_true
    cross_entp = K.sum(tf.multiply( GG, tf.log(tf.clip_by_value(y_pred,low_bound,1.0))), axis = -1)
    return tf.scalar_mul(-1,K.mean(cross_entp,axis=-1))


def cust_loss_critic(y_true, y_pred):
    
    R_t = y_true
    V_w = y_pred
    square_error = tf.square(tf.subtract(R_t, V_w))
    
    return K.mean(square_error,axis=-1)


def sample_from_policy(probabilities):
    
    elements = list(range(output_size))
    action = np.random.choice(elements, 1, p=probabilities) # sample from the distribution
    return action[0]



class A2C():
    

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        
        self.model = model
        #self.critic_model = critic_model
        self.n = n
        self.action_dim = 4
        self.gamma = gamma
        
        print('num_epochs,  lr, gamma:',num_epochs,lr_set,gamma)
        print('num_epochs2,  lr_critic, N:',num_epochs2,c_lr_set,self.n)

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.  
        self.inputs = Input(shape = (input_size ,))#input layer
        net = Dense(16, activation='relu',kernel_initializer='lecun_uniform')(self.inputs)#hidden 1
        net = Dense(16, activation='relu',kernel_initializer='lecun_uniform')(net)#hidden 2
        net = Dense(16, activation='relu',kernel_initializer='lecun_uniform')(net)#hidden3
        net = Dense(16, activation='relu',kernel_initializer='lecun_uniform')(net)#hidden3
        self.outputs =Dense(1, activation='linear',kernel_initializer='lecun_uniform')(net)#output layer
        
        self.critic_model = Model(inputs = self.inputs, outputs = self.outputs)
        
        
        
        
        self.model.compile(optimizer = Adam(lr = lr), loss = cust_loss_arctor) #'categorical_crossentropy')#cust_loss) 
        self.critic_model.compile(optimizer = Adam(lr = critic_lr), loss = 'mean_squared_error')#cust_loss_critic) #'categorical_crossentropy')#cust_loss) 

    def get_value(self, state):
        value = self.critic_model.predict(state.reshape(1,len(state)))
        return (value[0][0])

    def train(self, env, gamma=1.0):
        
        lr = lr_set
        flag1=True
        flag2=True
        
        evaluation = -1
        epi_count = 0
        recent_reward = []
        record_eval = []
        N = self.n
        Gap_count = 0
        
        
        
        while epi_count < 50000:
            epi_count +=1
            
            states, actions, rewards = self.generate_episode(env = env,is_train = True)
            
            T_sample = len(states)
            R_T = np.zeros((T_sample, output_size))
            R_T_critic = np.zeros(T_sample)
            V_W = []
            
            
            
            for i in list(range(T_sample)):
                input_state = np.reshape(states[i], (1, input_size))
                tmpt= self.critic_model.predict(x = input_state)
                V_W.append(tmpt[0][0])
            
            
            for t in list(reversed(range(T_sample))):
                V_end = 0 if (t + N >= T_sample) else V_W[t+N]
                tempR_T = (gamma**N) * V_end+ sum([(gamma**j) * rewards[t+j]/100.0 if (t+j<T_sample) else 0 for j in list(range(N))])
                R_T_critic[t]=tempR_T
                R_T[t, actions[t][0]] = tempR_T - V_W[t] #advantage value
                
                
                
            
            self.model.fit(x=np.array(states), y= R_T, epochs=num_epochs,batch_size=len(R_T),verbose=0)
            result = self.model.evaluate(x=np.array(states), y= R_T,verbose=0)
            self.critic_model.fit(x=np.array(states), y= R_T_critic, epochs=num_epochs2,batch_size=len(R_T),verbose=0)
            result_critic = self.critic_model.evaluate(x=np.array(states), y= R_T_critic,verbose=0)
            
            
            recent_reward.append(sum(rewards))
            if epi_count%1 ==0:
                action22 = self.model.predict(x = np.array(states))
                evaluation = sum(recent_reward)/len(recent_reward)
                print('{0},loss:{1}, critic liss:{2}, count:{3}'.format(evaluation,result,result_critic,epi_count))
                recent_reward=[]
            
            
            
            if epi_count%k ==0:
                mean, std = self.evaluate(env)
                print('episode # is {0},mean={1}, std={2}'.format(epi_count,mean,std))
                record_eval.append([epi_count,mean,std])
                print('num_epochs,  lr:',num_epochs,lr, low_bound)
                print('num_epochs2, lr_critic, N:',num_epochs2,c_lr_set,self.n)
                print(record_eval)
                self.model.save_weights("probm2_learned_weight.h5")
        
        
        
        return []
        
        
    
    
    
    
    
    
    
    
    def run_model(self, env, render=False):
        return self.generate_episode(env=env)
    
    
    def evaluate(self, env, render=False):
        reward_list =[]
        for i in list(range(100)):
            states, actions, reward = self.run_model(env = env)
            reward_list.append(sum(reward))
            
            
        
        mean = sum(reward_list)/100.0
        std = np.std(reward_list)
        
        return mean, std
    
    
    
    
    def generate_episode(self, env, render=False, is_train = False):
        # Generates an episode by executing the current policy in the given env.
        
        states = []
        actions = []
        rewards = []
        model = self.model
        
        
        state = env.reset()
        if render:
            env.render()
                        
        
        while True:
            input_state = np.reshape(state, (1, input_size))
            action = model.predict(x = input_state)
            action_value = action[0]
            action = np.argmax(action[0])
            
            
            if is_train:
                action = sample_from_policy(action_value)# initial action
                
            
            nextstate, reward, is_terminal, debug = env.step(action)
            
            states.append(np.reshape(state, (input_size,)))
            actions.append([action, action_value])
            rewards.append(reward)
            ###
            
            if render:
                env.render()
                
            state = nextstate
            
            if is_terminal:
                break

        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    
    
    env = gym.make('LunarLander-v2')
    
    
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    
    model2 = model
    a2c = A2C(model = model, lr = lr_set, critic_model = model2, critic_lr = c_lr_set, n = n)
    a2c .train(env, gamma=gamma)


if __name__ == '__main__':
    main(sys.argv)
