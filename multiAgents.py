# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#
# THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY
# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Artemis Kelly agkell3@emory.edu

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #My idea here was to calculate points for each move based on how close food was and
        #how far away the ghost is


        #we set the goal number from the given successor game state
        goal = successorGameState.getScore()
        #we set a list of food to be iterated through
        foodList = newFood.asList()

        #we want to consider the distance of every piece of food
        #we will add the reciprocal of the distance of the food because closer food is worth a higher value
        for food in foodList:
            distanceToFood = manhattanDistance(food, newPos)
            if (distanceToFood) != 0:
                goal += (1.0 / distanceToFood)

        #we next want to consider the distance of every ghost
        #we will add the reciprocal of the distance to the ghost because the score from the ghost will be negative
        #and the closer the ghost the higher the score should be
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            distanceToGhost = manhattanDistance(ghostPosition, newPos)
            if (abs(newPos[0] - ghostPosition[0]) + abs(newPos[1] - ghostPosition[1])) > 1:
                goal+= (1.0 / distanceToGhost)
        return goal

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        #I followed the algorithm we have covered in class with some changes to account for working with any
        #amount of ghosts

        #this function serves to choose if we are calling a max (pacman) agent or a min (ghost) agent
        def minimax(agentIndex, depth, gameState):
            #if we are at the final depth or the game is over we return
            if depth == self.depth or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            #if the agent is our pacman, we will call our max value function to choose his move
            if agentIndex == 0:
                max = maxValue(agentIndex, depth, gameState)
                return max
            #if the agent is a ghost, we will call our min value function to choose their moves
            if agentIndex >= 1:
                min = minValue(agentIndex, depth, gameState)
                return min

        def maxValue(agentIndex, depth, gameState):
            # we use an arbitrarily low number for min that will be used bc it is smaller than any potential return values
            max = -9999999
            #for each of the actions available, we run the evaluation function on the successor position and return
            #the best (highest value) position which we will choose
            #we evaluation by calling the minimax function recursively
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = minimax(1, depth, successor)
                #if our evaluation number is larger than the arbitrary number or the other successors values, set
                #max to that
                if value > max:
                    max = value
            return max

        def minValue(agentIndex, depth, gameState):
            #we use an arbitrarily high number for min that will be used bc it is larger than any potential return values
            min = 9999999

            nextIndex = agentIndex + 1
            #we have to update the index for eeach ghost
            if nextIndex <= 0 or gameState.getNumAgents() <= nextIndex:
                nextIndex = 0
                depth += 1
            # for each of the actions available, we run the evaluation function on the successor position and return
            # the correct (lowest value) position for the ghost
            # we evaluation by calling the minimax function recursively
            for x in gameState.getLegalActions(agentIndex):
                #evaluate each successor
                successor = gameState.generateSuccessor(agentIndex, x)
                value = minimax(nextIndex, depth, successor)

                #if the new value we found is less than our arbitrary number or one of the other
                #successor values, we return that
                if value < min:
                    min = value
            return min

        #set max and nextAction to default values
        max = -999999
        nextAction = Directions.NORTH

        #run the minimax function using the defined functions above
        for x in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, x)
            value = minimax(1, 0, successor)
            #if what we calculate is has a greater value than the arbitrary value or other successors we will
            #return the action associated with it
            if value > max:
                max = value
                nextAction = x

        return nextAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # if we are at the final depth or the game is over we return
            if depth == self.depth or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            #if the agent is our pacman, we will call our max value function to choose his move
            if agentIndex == 0:
                max = maxValue(agentIndex, depth, gameState, alpha, beta)
                return max
            # if the agent is a ghost, we will call our min value function to choose their moves
            if agentIndex >= 1:
                min = minValue(agentIndex, depth, gameState, alpha, beta)
                return min


        def minValue(agentIndex, depth, gameState, alpha, beta):
            # we use an arbitrarily high number for min that will be used bc it is larger than any potential return values
            min = 999999
            #we have to update the index for eeach ghost
            nextIndex = agentIndex + 1
            #if we have gone over the indexes
            if nextIndex <= 0 or gameState.getNumAgents() <= nextIndex:
                nextIndex = 0
                depth += 1
            # iterate through the actions, evaluating the value of each successor
            # use alpha and beta to try and avoid additional exploration of the tree
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = alphaBeta(nextIndex, depth, successor, alpha, beta)
                #check if this evaluation was better
                if value < min:
                    min = value
                #check if min is less than alpha
                if min < alpha:
                    return min
                #if beta is bigger, swap them
                if beta > min:
                    beta = min
            return min


        def maxValue(agentIndex, depth, gameState, alpha, beta):
            #set arbitrarily low value
            max = -999999
            #iterate through the actions, evaluating the value of each successor
            #use alpha and beta to try and avoid additional exploration of the tree
            for x in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, x)
                value = alphaBeta(1, depth, successor, alpha, beta)
                # check if this evaluation was better
                if value > max:
                    max = value
                #check if max is greater than beta in which case return rn
                if max > beta:
                    return max
                #check if alpha is smaller and swap if so
                if alpha < max:
                    alpha = max
            return max

        #set variables to arbitrary base values
        alpha = -999999
        beta = 999999
        max = -999999
        nextAction = Directions.NORTH

        #run the alphabeta equation
        for x in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, x)
            value = alphaBeta(1, 0, successor, alpha, beta)
            if value > max:
                nextAction = x
                max = value
            if value > beta:
                return value
            if alpha < value:
                alpha = value

        return nextAction
        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agentIndex, depth, gameState):
            # if we are at the final depth or the game is over we return
            if depth == self.depth or (gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)

            #if the agent is our pacman, we will call our max value function to choose his move
            if agentIndex == 0:
                max = maxValue(agentIndex, depth, gameState)
                return max
            #if the agent is a ghost, we will call our min value function to choose their move
            if agentIndex >= 1:
                min = minValue(agentIndex, depth, gameState)
                return min

        #same as above maxValue functions
        def maxValue(agentIndex, depth, gameState):
            max = -999999
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = expectimax(1, depth, successor)
                if value > max:
                    max = value
            return max

        def minValue(agentIndex, depth, gameState):
            #we will use the average of the ghost values as the min because they choose
            #values randomly
            #this will represent the total of the values of all ghosts
            total = 0
            #this will be the amount of ghosts
            amount = 0
            #same as above
            nextIndex = agentIndex + 1
            if nextIndex <= 0 or gameState.getNumAgents() <= nextIndex:
                nextIndex = 0
                depth += 1
            #similar to above except we are keeping track of all of the ghosts values
            #and the amount of ghost values to allow us to take an average
            for x in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, x)
                value = expectimax(nextIndex, depth, successor)
                total += value
                amount += 1
            return total / amount

        #reset values
        max = -999999
        nextAction = Directions.NORTH
        for x in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(self.index, x)
            value = expectimax(1, 0, successor)
            if value > max:
                max = value
                nextAction = x

        return nextAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I got the location of all of the ghosts and food and how many super pellets there were. I calculated
      the minimum distance to ghosts and scared ghosts and then added points accordingly to guide pacman to
      avoid ghosts that were close. I followed the prompt and used negative numbers.

    """
    "*** YOUR CODE HERE ***"

    #values I will be referencing to create a weighted point system
    ghostList = currentGameState.getGhostStates()
    currentPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood()
    superPellets = currentGameState.getCapsules()

    #variables whose value I will be changing to be the multiplication factor later
    distanceToGhosts = []
    minDistanceToGhosts = -1
    distanceToScaredGhosts = []
    minDistanceToScaredGhosts = -1

    #return if we are in winning game state or losing game state
    if currentGameState.isLose():
        return float("-999999")
    if currentGameState.isWin():
        return float("999999")

    # get the distance to each ghost and each scared ghost
    for x in ghostList:
        if x.scaredTimer == 0:
            distanceToGhosts.append(manhattanDistance(currentPosition, x.getPosition()))
        elif x.scaredTimer > 0:
            distanceToScaredGhosts.append(manhattanDistance(currentPosition, x.getPosition()))

    #find the minimum distance to each ghost and each scared ghost
    if len(distanceToGhosts) > 0:
        minDistanceToGhosts = min(distanceToGhosts)
    if len(distanceToScaredGhosts) > 0:
        minDistanceToScaredGhosts = min(distanceToScaredGhosts)
    #get the distance to each food
    distanceToFoods = []
    for x in foodList.asList():
        distanceToFoods.append(manhattanDistance(x, currentPosition))
    #get min distance to foods
    minDistanceToFoods = min(distanceToFoods)


    #start with score that is calculated from the general evaluation function
    score = scoreEvaluationFunction(currentGameState)
    score += (-20 * len(superPellets))

    score += (-8 * len(foodList.asList()))
    score += (-3 * minDistanceToFoods)

    score += (-4 * (1.0/minDistanceToGhosts))
    score += (-4 * (1.0/minDistanceToScaredGhosts))

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        #my goal for this algorithm is to try and get a relatively high score even if pacman is going to die
        #im implementing concepts from alpha beta pruning and minimax to calculate what pacman's best move is
        #and what is the most likely next move for each of the smart ghosts

        answer = self.minicontest(gameState, 0, 1, -9999999, 9999999)
        return answer[0]
        util.raiseNotDefined()

    def minicontest(self, gameState, agent, depth, alpha, beta):
        #default is to set this equal to alpha if its the pacman because we are going for max value
        nextAction = (Directions.STOP, alpha);

        #if its not pacman we set to beta bc we are going for min value
        if agent != 0:
            nextAction = (Directions.STOP, beta);

        # return if we are in winning game state or losing game state or at depth
        #return the score from our better eval function
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return (Directions.STOP, betterEvaluationFunction(gameState));

        #get all legal actions from current spot
        possibleActions = gameState.getLegalActions(agent)

        for x in possibleActions:
            #iterate through the list of possible actions
            action = x
            #if action is not to stop, calculate a new player
            if action != Directions.STOP:
                #incase the current agent is the last ghost so it wraps around indexes
                agent2 = (agent + 1) % gameState.getNumAgents()
                depth2 = depth
                gameState2 = gameState.generateSuccessor(agent, action)

                #check if agent is pacman in which case increase the depth for the evaluation
                if agent2 == 0:
                    depth2 += 1

                value = self.minicontest(gameState2, agent2, depth2, alpha, beta)

                #if the agent is not pacman and the valculated value is less than the value of the nextAction
                #or if the agent is pacman and the value is greater than the nextAction change the value
                #of next action to the improved value/move
                if (agent != 0 and value[1] < nextAction[1]) or (agent == 0 and value[1] > nextAction[1]):
                    nextAction = (action, value[1])

        return nextAction
