from gridWorld import gridWorld
import numpy as np
import random
discount = 0.99
actions = 4
states = 17
thresh = 0.01
Actions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]

#Inputs: Transition Matrix T, Rewards Matrix R, state s, discount factor gamma
#Outputs (a*, V(s)) Optimal Action a and expected value s
def getV(T, R, gamma, V=None, P=None, V_p=None):
    Q = np.zeros((states, actions))
    if P is None:
        P = np.zeros((states))
    if V is None:
        V = np.zeros((states))

    #get set of all reachable states
    for state in range(states):
        #V_p = np.copy(V)
        #print("Reachable States:", next_states)        
        for act in range(actions):
            sp_lst = np.where(T[state, :, act] > 0)
            next_states = np.unique(sp_lst[0])
            
            #get probability of all reachable states given an action
            pn_states = T[state, next_states, act]
            #print("P(s' | ",state, ",", act, ")=", pn_states)
            #Q[state, act] = R[state] + gamma * np.sum(pn_states * V_p[next_states])
            Q[state, act] = R[state] + gamma * pn_states.dot(V_p[next_states])
            #if state == 8:
                #print("Q(", state,",",act,")=", Q[state,act])
        if state == 16:
            V[state] = R[state]
            P[state] = 4
        else:
            V[state] = np.max(Q[state,:]) 
            if state == 8:
                print("Q_max", np.max(Q[state,:]))
            P[state] = np.argmax(Q[state,:])
            
    #print("V:", V)    
    #print("P:", P)
    return V,P.astype(int)

def passThreshold(V, V2, threshold):
    V_diff = abs(V - V2)
    V_less = V_diff[V_diff < threshold]
    return (V_less.size == V_diff.size)

def ValIterate(T, R, threshold, gamma):
    V = np.zeros((states))
    P = np.zeros((states))
    i = 0
    while True:
        V2 = np.copy(V)
        getV(T,R, gamma, V, P, V2)
        if passThreshold(V, V2, threshold):
            print("Converged after" , i ,  "iterations")
            break
        i+=1
    return V,P.astype(int)

def printPolicy(Policy):
    i = 0
    for p in Policy:
        print("State", i , ":" ,  Actions[p])
        i+=1
        
def printValue(Value):
    i = 0
    for v in Value:
        print("V(", i , ")=" ,  np.around(v, decimals=3))
        i+=1

#returns 1/N(s,a)
def UpdateAlpha(A, state, action):
    A[state, action] +=1
    return 1/A[state, action]

def ExecuteAction2(T, state, action, p):
    a = p
    b = np.around((1 - p)/2, 2)
    rand = random.random()
    sp_lst = np.where(T[state, :, action] > 0)
    subindex = 0
    next_states = np.unique(sp_lst[0])
    if p == 1 or T[state, next_states[0], action] == 1:
        return next_states[0]

    if next_states.size == 2:
        if rand <= a + b:
            index = np.where(T[state, next_states, action] == a + b)[0]
        else:
            index = np.where(T[state, next_states, action] == b)[0]
    elif next_states.size == 3:
        if rand <= a:
            index = np.where(T[state, next_states, action] == a)[0]
        elif rand <= a + b:
            index = np.where(T[state, next_states, action] == b)[0] 
            subindex = 0
        else:
            index = np.where(T[state, next_states, action] == b)[0]
            subindex = 1
            
    next_state = next_states[index]
    
    if next_state.size is not 0:
        return next_state[subindex]
    else:
        print("Rand", rand, next_states.size, b,T[state, next_states, action])
        return next_states[0]
    
def ExecuteAction(T, state, action, p):
    a = p
    b = np.around((1 - p)/2, 2)
    rand = random.random()
    sp_lst = np.where(T[state, :, action] > 0)
    next_states = np.unique(sp_lst[0])
    if p == 1 or T[state, next_states[0], action] == 1:
        return next_states[0]
    
    #print("N_states:", next_states)
    #print("Probability:", T[state, next_states, action])
    #print("Random Value:", rand)
    if rand <= a:
        index = np.where(T[state, next_states, action] >= a)[0]
    elif rand <= a + b:
        index = np.where(T[state, next_states, action] == a + b)[0]
        if index.size == 0:
            index = np.where(T[state, next_states, action] == b)[0]
    elif rand <=b:
        index = np.where(T[state, next_states, action] == b)[0]
    else:
        index = np.where(T[state, next_states, action] <= b)[0]
        
    next_state = next_states[index]
    #print(index)
    if next_state.size > 1:
        return next_state[random.randint(0, next_state.size - 1)]
    elif next_state.size > 0:
        return next_state[0]
    else: 
        return state

def QLearn(R,T, alpha, gamma, epnum, epsilon):
    Q = np.zeros((states, actions))
    V = -np.ones((states, actions))
    A = np.zeros((states, actions))
    acts = np.array(range(actions))
    #set the start state
    s_0 = 4
    goal = 15
    end = 16
    for i in range(epnum):
        #reset state to the initial state
        state = s_0
        done = False
        while not done:
            #select an action
            #epsilon greedy
            if random.random() > epsilon:
                #select the optimal action
                next_act = np.argmax(Q[state,:])
            else:
                #select a random action
                np.random.shuffle(acts)
                next_act = acts[0]
             
            #print("Best Action in S:",state, "is:", Actions[next_act.astype(int)])          
            #execute it
            act = next_act
            alpha = UpdateAlpha(A, state, act)
            nstate = ExecuteAction2(T, state, act, 0.9) 
            
            Q_prev = Q[state, act]
            #print("Q_prev(", state, ",", Actions[act],") ="  , Q_prev)
            #print("max Q(", nstate, ",a') ="  , np.max(Q[nstate, :]))
            Q_next = Q_prev + alpha *(R[state] + gamma * np.max(Q[nstate, :]) - Q_prev)
            V[nstate, act] = np.max(Q[nstate, :]) - Q_prev
            Q[state, act] = Q_next
            #print("Q(", state, ",", Actions[act],")="  , Q_next)
            
            if nstate == end:
                done = True
            if nstate is not None:
                state = nstate
            else:
                print("Error chose NULL state!")
                
            #print("s'=" , nstate)
            #print("Alpha: ", alpha)
            
           
        #print("Goal Reached!", i, "episode completed!")
    print("Learning Complete!")
    return Q

def QBellman(Q_opt, T, R):
    gamma = discount
    Q_diff = np.zeros((states, actions))
    Q_real = np.zeros((states, actions))
    for state in range(states):
        reward = R[state]
        for act in range(actions):
            sp_lst = np.where(T[state, :, act] > 0)
            next_states = np.unique(sp_lst[0])            
            Q = Q_opt[state, act]
            p = T[state, next_states,act]
            #Q(s',a')
            Q_sa = Q_opt[next_states, :]
            #print("Q(s', a')=", Q_sa)
            Q_max = np.max(Q_sa, axis = 1)
            #print("QMax=",Q_max)
            #print("P( s', s, a):", p)
            pq_max = p * Q_max
            #print("P*QMax=", pq_max)
            Q_t = reward + gamma * np.sum(pq_max)
            #print("Q_approx", Q)
            #print("Q_actual:", Q_t)
            Q_diff[state, act] = abs(Q_t - Q)
            Q_real[state,act] = Q_t
        
    Qflat = Q_diff.flatten()
    
    return np.mean(Qflat)

def Bellman(V_opt, T, R):
    gamma = discount
    V_diff = np.zeros((states))
    V_real = np.zeros((states))
    V_acts = np.zeros((actions))
    for state in range(states):
        reward = R[state]
        V = V_opt[state]
        for act in range(actions):
            sp_lst = np.where(T[state, :, act] > 0)
            next_states = np.unique(sp_lst[0])            
            p = T[state,next_states,act]
            
            #V(s')
            V_sa = V_opt[next_states]
            
            pv_sa = p * V_sa
            
            #print("P*QMax=", pq_max)
            V_acts[act] = reward + gamma * np.sum(pv_sa)
    
        V_t = np.max(V_acts)
        V_diff[state] = abs(V_t - V)
        V_real[state] = V_t
        
    print("V_opt:", V_real)
    print("V_actual", V_opt)
    
    Vflat = V_diff.flatten()
    
    return np.mean(Vflat)

def getOptPolicy(Q):
    A = np.zeros((states))
    for state in range(states):
        A[state] = np.argmax(Q[state,:])
        
    return A.astype(int)

def getOptVal(Q):
    V = np.zeros((states))
    for state in range(states):
        V[state] = np.max(Q[state,:])
    return V

def main():
    TR = gridWorld()
    T = TR[0]
    R = TR[1]
    V1 = np.zeros((8))
    V2 = np.ones((8))
    itr = 10000
    alpha = 1
    Resu =  ValIterate(T, R, thresh, discount)
    V_opt = Resu[0]
    P_opt = Resu[1]
    #print("Diff", Bellman(V_opt, T, R))
    #print("V*:", V_opt)
    #print("P*:", P_opt)
    #print("Optimal Policy")
    #printPolicy(P_opt)
    #print("Optimal Value Function")
    #printValue(V_opt)
   
    Q_opt = QLearn(R, T , alpha, 0.95, itr, 0.05)
    
    Q_opt2 = QLearn(R, T , alpha, 0.99, itr, 0.2)
    
    print("e=0.05", QBellman(Q_opt, T, R))
    P_opt = getOptPolicy(Q_opt)
    V_opt = getOptVal(Q_opt)
    
    P_opt2 = getOptPolicy(Q_opt2)
    V_opt2 = getOptVal(Q_opt2)        
    
    printValue(V_opt)
    printPolicy(P_opt)
    
    print("e=0.2",QBellman(Q_opt2, T, R))
    printValue(V_opt2)
    printPolicy(P_opt2)    
    return 0

if __name__ == "__main__":
    main()