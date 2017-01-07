import random
from Dictionary import Dictionary


class UFSACO:
    def __init__(self,
                 numberAnts,
                 numberFeatures,
                 dictionaryName,
                 dictionaryFolderHier='',
                 numberCycles=50,
                 decayRate=0.2,
                 beta=1,
                 initialPheromone=0.2,
                 exploreExploitCoeff=0.7
                 ):
        """
        UFSACO algorithm to find optimal feature subset based on unsupervised classification
        and Ant Colony Optimization
        :param numberAnts: Number of ants (agents)
        :param numberFeatures: Number of features to extract. Must be lower than number of features of dictionary
        :param dictionaryName: Name of dictionary to load
        :param dictionaryFolderHier: Hierarchy of dictionary object (for routing)
        :param numberCycles: Number of times the ants must search through feature space
        :param decayRate: Pheromone decay rate [0 to 1]
        :param beta: Beta value for transition rule
        :param initialPheromone: Initial pheromone value
        :param exploreExploitCoeff: Exploration / exploitation coefficient [0 to 1]
        """
        # Initialize posting tokens
        self.postingTokens = set()

        # Attempt to load dictionary
        self.dictionary = Dictionary(dictionaryName=dictionaryName, folderHierarchy=dictionaryFolderHier)
        self.dictExists = self.dictionary.loadFromDisk()

        # Load dictionary similarities
        if self.dictExists is True:
            # Load similarities
            self.dictionary.loadSimilarities()

            # Keep dictionary postings as a set
            self.postingTokens = set(self.dictionary.postings)

            # Set parameters for algorithm
            self.numberCycles = numberCycles
            self.numberAnts = numberAnts
            self.beta = beta
            self.initialPheromone = initialPheromone

            # Verify decay rate between 0 and 1
            if 0 <= decayRate <= 1:
                self.decayRate = decayRate
            else:
                self.decayRate = 0.2

            # Verify exploration/exploitation coefficient between 0 and 1
            if 0 <= exploreExploitCoeff <= 1:
                self.exploreExploitCoefficient = exploreExploitCoeff
            else:
                self.exploreExploitCoefficient = 0.7

            # Verify number of terms in dictionary to set selected number of features
            if self.dictionary.termCount > numberFeatures:
                self.numberFeatures = numberFeatures
            else:
                self.numberFeatures = self.dictionary.termCount

            # List of feature selection counter for each iteration
            self.featureCounterIteration = {}

            # Initialize feature counter variable for ants
            self.featureCounter = {}
            self.totalFeatureCounter = 0

    def initPheromone(self):
        """
        Initialize pheromone value for all features
        :return:
        """
        print '[Initializing pheromone values]'
        self.pheromoneValue = {}

        for token in self.postingTokens:
            self.pheromoneValue[token] = self.initialPheromone

    def updatePheromone(self, cycleIteration):
        """
        Global pheromone update
        :param cycleIteration: Iteration of pheromone update
        :return:
        """
        if self.dictExists is True:
            # Update pheromone in each token
            decayValue = 1 - self.decayRate

            for token in self.postingTokens:
                tokenPheromoneValue = decayValue * self.pheromoneValue[token]

                if self.totalFeatureCounter > 0 and token in self.featureCounter:
                    tokenPheromoneValue += float(self.featureCounter[token]) / self.totalFeatureCounter

                # Assign new value to pheromone
                self.pheromoneValue[token] = tokenPheromoneValue

            # Store feature counter for iteration
            self.featureCounterIteration[cycleIteration] = self.featureCounter

    def getUnvisitedHeuristics(self, currentToken, unvisitedTokenList):
        """
        Calculate heuristics information for unvisited feature list
        when ant is in a specific feature
        :param currentToken: Ant position
        :param unvisitedTokenList: List of unvisited features by ant
        :return: List [heuristics: list of heuristic values, max_token: for greedy movement, total_heuristics: sum of heuristic values]
        """
        # Keep token having maximum value of heuristics
        argMaxValue = 0
        argMaxToken = None

        # List of unvisited heuristics
        unvisitedHeuristics = {}

        # Keep total of heuristics sum
        totalHeuristics = 0

        for unvisitedToken in unvisitedTokenList:
            heuristicsValue = float(self.pheromoneValue[unvisitedToken] * (
                self.dictionary.getSimilarity(currentToken, unvisitedToken) ** self.beta))

            # Store heuristics value if it is different from zero
            if heuristicsValue != 0:
                unvisitedHeuristics[unvisitedToken] = heuristicsValue

                # Increment total of heuristics
                totalHeuristics += heuristicsValue

                # In case obtained value is higher than previous maximum, substitute
                if heuristicsValue > argMaxValue:
                    argMaxValue = heuristicsValue
                    argMaxToken = unvisitedToken

        return {'heuristics': unvisitedHeuristics, 'max_token': argMaxToken, 'total_heuristics': totalHeuristics}

    def probabilityTransitionRule(self, unvisitedHeuristics, totalUnvisited):
        """
        Get token with highest probability to be chosen as next feature
        :param unvisitedHeuristics: List of unvisited tokens heuristics
        :param totalUnvisited: Total of sum of tokens heuristics
        :return: Token with highest probability
        """
        argMaxProbValue = 0
        argMaxToken = None
        for unvisitedToken in unvisitedHeuristics:
            probValue = unvisitedHeuristics[unvisitedToken] / totalUnvisited

            # In case obtained value is higher than previous maximum, substitute
            if probValue > argMaxProbValue:
                argMaxProbValue = probValue
                argMaxToken = unvisitedToken

        return argMaxToken

    def moveAnt(self, currentFeature):
        """
        Move ant to next feature and modify feature counter
        :param currentFeature: Feature where ant has been positioned
        :return:
        """
        # Initialize unvisited features
        unvisitedFeatureList = self.postingTokens.difference(currentFeature)

        # Execute according to the number of features an ant has to move in
        for featureNumber in range(0, self.numberFeatures):
            # Get heuristic information for unvisited features
            heuristicsInformation = self.getUnvisitedHeuristics(
                currentToken=currentFeature,
                unvisitedTokenList=unvisitedFeatureList
            )

            """
            Random assignment to choose transition rule [0, 1]
            On each feature, ant decides to move in a greedy or probabilistic way
            """
            transitionSelection = random.random()

            if transitionSelection <= self.exploreExploitCoefficient:
                # Get next feature from maximum value found in heuristics information
                nextFeature = heuristicsInformation['max_token']
            else:
                # Calculate transition using probabilistic rule
                nextFeature = self.probabilityTransitionRule(unvisitedHeuristics=heuristicsInformation['heuristics'],
                                                             totalUnvisited=heuristicsInformation['total_heuristics'])

            # Move ant to new feature
            if nextFeature is not None:
                currentFeature = nextFeature

                # Remove feature from unvisited list
                unvisitedFeatureList.remove(nextFeature)

                # Update counter
                if nextFeature not in self.featureCounter:
                    self.featureCounter[nextFeature] = 0

                self.featureCounter[nextFeature] += 1
                self.totalFeatureCounter += 1
            else:
                break

    def searchSubset(self):
        """
        Perform ACO algorithm for searching subset of features
        :return:
        """
        if self.dictExists is True:
            # Step 1: initialize pheromone
            self.initPheromone()

            # Get term count range from dictionary
            termCountRange = self.dictionary.termCount - 1

            # Execute searching for a number of iterations set in the constructor
            cycleIteration = 0

            # Create range for ants
            antRange = range(0, self.numberAnts)

            # This part will be executed self.numberCycles times from the constructor
            while cycleIteration < self.numberCycles:
                print '[Iteration #' + str(cycleIteration + 1) + ']'

                # Step 2: place ants in random features
                self.featureCounter = {}  # Initialize feature counter in each iteration
                self.totalFeatureCounter = 0  # Initialize total feature counter

                # This vector is used to assign ants in different features randomly
                initialFeaturesValues = []
                for antNumber in antRange:
                    # Assign a unique feature to each ant
                    while True:
                        randomFeatureValue = random.randint(0, termCountRange)
                        if randomFeatureValue not in initialFeaturesValues:
                            break

                    # Assign feature to ant and save into list of visited
                    antCurrentFeature = self.dictionary.postings[randomFeatureValue]

                    # Append feature value to list
                    initialFeaturesValues.append(randomFeatureValue)

                    # Create new thread, append to list
                    self.moveAnt(currentFeature=antCurrentFeature)

                # Step 4: global pheromone update
                self.updatePheromone(cycleIteration=cycleIteration)

                # Add iteration counter
                cycleIteration += 1

    def getFeatureResults(self, topNumber, onlyTokens=True):
        """
        Return top m features after searching subset
        :param topNumber: Top number of features to retrieve
        :param onlyTokens: Get only token list. If False, return also pheromone value
        :return:
        """
        # Sort results based on pheromone value
        orderedFeatures = sorted(self.pheromoneValue, key=self.pheromoneValue.__getitem__, reverse=True)

        if len(orderedFeatures) >= topNumber:
            orderedFeatures = orderedFeatures[0:topNumber]

        # Return only token list
        if onlyTokens is True:
            return orderedFeatures
        else:
            # Return top m features
            featureResults = {}
            for token in orderedFeatures:
                featureResults[token] = self.pheromoneValue[token]

            return featureResults
