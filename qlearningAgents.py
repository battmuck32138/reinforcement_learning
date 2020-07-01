# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *


import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # can play around with the knobs but it screws up autograder
        # because optimal setting change from task to task 
        # set up for the default settings.
        #self.epsilon = 0.5    # exploration probe
        #self.alpha = 0.2   # learning rate
        #self.gamma = 0.8    # discount
        self.qvalues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state, action)]


    def computeValueFromQValues(self, state):
        """ this is max_a Q(s, a)
          Returns max_action Q(state, action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)

        if legal_actions:
            return max([self.getQValue(state, a) for a in legal_actions])

        else:
            return 0.0


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
          Returns an actions
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return None

        action_QValue_pairs = [(a, self.getQValue(state, a)) for a in legal_actions]
        max_pair = max(action_QValue_pairs, key=lambda x: x[1])
        return max_pair[0]


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        random_action = util.flipCoin(self.epsilon)

        if not legal_actions:
            return None

        if random_action:
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # SAMPLE:  sample = R(s, a, s') + gamma max_a' Q(s', a')
        # UPDATE:  Q(s, a) = (1 - alpha) Q(s, a) + alpha [sample]
        R = reward
        max_Q = self.getValue(nextState)
        gamma = self.discount
        Q = self.qvalues[(state, action)]

        sample = R + gamma * max_Q
        self.qvalues[(state, action)] = (1 - self.alpha) * Q + self.alpha * sample


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)


    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = np.dot(w, feature_vector)
        import numpy as np
        w = np.array(self.getWeights())
        feature_vector = np.array(self.featExtractor.getFeatures(state, action))
        return np.dot(w, feature_vector)


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # difference = (r + gamma * max_a'  Q(s', a')) - Q(s, a)
        # UPDATE:  w += alpha * difference * f(s, a)
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        feature_vector = self.featExtractor.getFeatures(state, action)

        for feature in feature_vector:
            self.weights[feature] += self.alpha * difference * feature_vector[feature]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
