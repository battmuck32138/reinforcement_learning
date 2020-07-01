# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        """
        terminal states have no reward, set val to 0
        self.values is a dictionary of {s : q_val, ....}
        I want to update the values associated with s

         """
        "*** YOUR CODE HERE ***"

        # k iterations
        for _ in range(self.iterations):
            update_values_dict = self.values.copy()

            # get Q_values for each possible s_prime
            for state in self.mdp.getStates():
                Q_values = [float('-inf')]
                terminal_state = self.mdp.isTerminal(state)  # boolean

                # Terminal states have 0 value.
                if terminal_state:
                    update_values_dict[state] = 0

                else:
                    legal_actions = self.mdp.getPossibleActions(state)

                    for action in legal_actions:
                        Q_values.append(self.getQValue(state, action))

                    # update value function at state s to largest Q_value
                    update_values_dict[state] = max(Q_values)

            self.values = update_values_dict


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).

        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          Q-Values: expected future utility from a q-state (chance node)
          Q*(s, a) = sum_s' of T(s, a, s') [R(s, a, s') + gamma V*(s')]
        """
        "*** YOUR CODE HERE ***"
        possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)  # [(s_prime, prob), ...]
        value_utilities = []

        for s_prime, T in possible_next_states:
            R = self.mdp.getReward(state, action, s_prime)
            gamma_V = self.discount * self.values[s_prime]
            value_utilities.append(T * (R + gamma_V))

        return sum(value_utilities)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          This is policy Extraction
          V*(s) = max_a Q*(s, a)
          return the action with the highest Q-value
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.mdp.getPossibleActions(state)

        if len(legal_actions) == 0:
            return None

        actions_qvals = []  # [(action, qval) ... ]

        for action in legal_actions:
            qval = self.getQValue(state, action)
            actions_qvals.append((action, qval))

        action_with_best_qval = max(actions_qvals, key=lambda x: x[1])[0]
        return action_with_best_qval


    def getPolicy(self, state):
        return self.computeActionFromValues(state)


    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)


    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)



class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        # initialize value function to all 0 values.
        for state in states:
            self.values[state] = 0

        num_states = len(states)

        for i in range(self.iterations):
            state_index = i % num_states
            state = states[state_index]

            terminal = self.mdp.isTerminal(state)
            if not terminal:
                action = self.getAction(state)
                qval = self.getQValue(state, action)
                self.values[state] = qval


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.runValueIteration()


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        fringe = util.PriorityQueue()
        predecessors = {}

        for s in states:
            self.values[s] = 0
            predecessors[s] = self.get_predecessors(s)

        for s in states:
            terminal = self.mdp.isTerminal(s)

            if not terminal:
                current_value_of_state = self.values[s]
                diff = abs(current_value_of_state - self.max_Qvalue(s))
                fringe.push(s, -diff)

        for _ in range(self.iterations):

            if fringe.isEmpty():
                return

            s = fringe.pop()
            self.values[s] = self.max_Qvalue(s)

            for p in predecessors[s]:
                diff = abs(self.values[p] - self.max_Qvalue(p))
                if diff > self.theta:
                    fringe.update(p, -diff)


    def max_Qvalue(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])


    # First, we define the predecessors of a state s as all states that have
    # a nonzero probability of reaching s by taking some action a
    # This means no Terminal states and T > 0.
    def get_predecessors(self, state):
        predecessor_set = set()
        states = self.mdp.getStates()
        movements = ['north', 'south', 'east', 'west']

        if not self.mdp.isTerminal(state):

            for p in states:
                terminal = self.mdp.isTerminal(p)
                legal_actions = self.mdp.getPossibleActions(p)

                if not terminal:

                    for move in movements:

                        if move in legal_actions:
                            transition = self.mdp.getTransitionStatesAndProbs(p, move)

                            for s_prime, T in transition:
                                if (s_prime == state) and (T > 0):
                                    predecessor_set.add(p)

        return predecessor_set




