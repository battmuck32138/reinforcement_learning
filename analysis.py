# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    """
    Only want agent to survive for 3 steps so big living penalty
    I don't want agent to fear fire so no noise
    I want agent to move toward small reward so discount the 10
    """
    answerDiscount = 0.1
    answerNoise = 0
    answerLivingReward = -4.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
    Only want agent to survive 7 steps
    Want agent to be scared of fire so add some noise
    """
    answerDiscount = 0.1
    answerNoise = .1
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
    I want agent to live 5 steps
    I don't want agent ot fear the fire so no noise
    """
    answerDiscount = 1
    answerNoise = 0.0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
    I want to avoid cliff so add noise
    I want distant reward of 10 so discount alpha = 1
    I want agent to live at least 10 steps so small living penalty
    """
    answerDiscount = 1
    answerNoise = .1
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
    I want agent to live forever so living Reward is big
    I want agent to avoid cliff so add noise
    As long as living reward is > 10, no need to discount.
    """
    answerDiscount = 1
    answerNoise = 0.1
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    #return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
