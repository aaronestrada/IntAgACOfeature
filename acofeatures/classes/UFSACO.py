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

    def initPheromone(self):
        """
        Initialize pheromone value for all features
        :return:
        """
        print 'Initializing pheromone values...'
        self.pheromoneValue = {}

        for token in self.postingTokens:
            self.pheromoneValue[token] = self.initialPheromone

    def updatePheromone(self, featureCounter):
        """
        Global pheromone update
        :param featureCounter: List of feature counter for each iteration in the ACO process
        :return:
        """
        print 'Updating pheromone values...'
        if self.dictExists is True:
            # Calculate total of feature counter movements
            totalFeatureCounter = 0
            for token in featureCounter:
                totalFeatureCounter += featureCounter[token]

            # Update pheromone in each token
            for token in self.postingTokens:
                tokenPheromoneValue = (1 - self.decayRate) * self.pheromoneValue[token]

                if totalFeatureCounter > 0 and token in featureCounter:
                    tokenPheromoneValue += float(featureCounter[token]) / totalFeatureCounter

                # Assign new value to pheromone
                self.pheromoneValue[token] = tokenPheromoneValue

    def greedyTransitionRule(self, token1, unvisitedTokenList):
        """
        Calculate greedy transition rule
        :param token1: Token (from) to verify transition rule
        :param unvisitedTokenList: List of unvisited tokens
        :return: List with value and token to move in
        """
        argMaxValue = 0
        argMaxToken = None
        for unvisitedToken in unvisitedTokenList:
            totalValue = self.pheromoneValue[unvisitedToken] * (self.dictionary.getSimilarity(token1,
                                                                                              unvisitedToken)) ** self.beta

            # In case obtained value is higher than previous maximum, substitute
            if totalValue > argMaxValue:
                argMaxValue = totalValue
                argMaxToken = unvisitedToken

        return argMaxToken

    def getTotalTransitionRule(self, token, unvisitedTokenList):
        """
        Get total of transition rule among a list of features
        :param unvisitedTokenList: List of features to calculate similarities with
        :return: Total calculated value
        """
        totalUnvisited = 0
        for unvisitedToken in unvisitedTokenList:
            totalUnvisited += self.pheromoneValue[unvisitedToken] * (
                (self.dictionary.getSimilarity(token, unvisitedToken)) ** self.beta)
        return totalUnvisited

    def probabilityTransitionRule(self, token1, token2, totalUnvisited):
        """
        Calculate probability transition rule between tokens
        :param token1: Token (from) to verify transition rule
        :param token2: Token (to) to verify transition rule
        :param totalUnvisited: Total of transition rule among all features
        :return: Probability transition rule
        """
        return float(
            self.pheromoneValue[token2] * (self.dictionary.getSimilarity(token1, token2) ** self.beta)) / totalUnvisited

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
            cycleIteration = self.numberCycles

            # Create range for ants
            antRange = range(0, self.numberAnts)

            # This part will be executed self.numberCycles times from the constructor
            while cycleIteration > 0:
                # Step 2: place ants in random features
                ants = {}  # Initialize ants in each iteration
                featureCounter = {}  # Initialize feature counter in each iteration
                visitedFeatures = {}

                # This vector is used to assign ants in different features randomly
                initialFeaturesValues = []
                for antNumber in antRange:
                    # Assign a unique feature to each ant
                    while True:
                        randomFeatureValue = random.randint(0, termCountRange)
                        if randomFeatureValue not in initialFeaturesValues:
                            break

                    # Assign feature to ant and save into list of visited
                    ants[antNumber] = self.dictionary.postings[randomFeatureValue]

                    # initialize visited features
                    visitedFeatures[antNumber] = []

                    # Append feature value to list
                    initialFeaturesValues.append(randomFeatureValue)

                # Execute according to the number of features each ant has to move in
                for featureNumber in range(0, self.numberFeatures):
                    print 'Selecting feature ' + str(featureNumber) + '...'
                    # Step 3: Move ants to the next feature
                    for antNumber in antRange:
                        currentFeature = ants[antNumber]

                        # Get list of unvisited features
                        unvisitedFeatureList = self.postingTokens.difference(visitedFeatures[antNumber])

                        # Random assignment to choose transition rule [0, 1]
                        transitionSelection = random.uniform(0, 1)

                        if transitionSelection <= self.exploreExploitCoefficient:
                            # Calculate transition using greedy rule and assign next feature
                            print 'Moving ant (greedy) ' + str(antNumber) + '...'
                            nextFeature = self.greedyTransitionRule(token1=currentFeature,
                                                                    unvisitedTokenList=unvisitedFeatureList)
                        else:
                            print 'Moving ant (probabilistic) ' + str(antNumber) + '...'
                            # Calculate transition using probabilistic rule
                            totalUnvisited = self.getTotalTransitionRule(currentFeature,
                                                                         unvisitedTokenList=unvisitedFeatureList)

                            if totalUnvisited > 0:
                                argMaxProbValue = 0
                                # Calculate probability of moving to the next feature
                                for possibleNextToken in unvisitedFeatureList:
                                    probValue = self.probabilityTransitionRule(token1=currentFeature,
                                                                               token2=possibleNextToken,
                                                                               totalUnvisited=totalUnvisited)

                                    # In case a higher probability has been found, substitute it
                                    if probValue > argMaxProbValue:
                                        argMaxProbValue = probValue
                                        nextFeature = possibleNextToken

                        # Move ant to new feature
                        ants[antNumber] = nextFeature

                        # Add feature to visited list
                        visitedFeatures[antNumber].append(nextFeature)

                        # Update counter
                        if nextFeature not in featureCounter:
                            featureCounter[nextFeature] = 0
                        featureCounter[nextFeature] += 1

                print featureCounter

                # Step 4: global pheromone update
                self.updatePheromone(featureCounter=featureCounter)

                # Subtract iteration counter
                cycleIteration -= 1

    def getFeatureResults(self, topNumber):
        """
        Return top m features after searching subset
        :param topNumber: Top number of features to retrieve
        :return:
        """
        featureResults = {}

        # Sort results based on pheromone value
        orderedFeatures = sorted(self.pheromoneValue, key=self.pheromoneValue.__getitem__, reverse=True)

        if len(orderedFeatures) >= topNumber:
            orderedFeatures = orderedFeatures[0:topNumber - 1]

        # Return top m features
        for token in orderedFeatures:
            featureResults[token] = self.pheromoneValue[token]

        return featureResults
