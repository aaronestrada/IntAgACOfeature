import os
import sys
import json
import argparse
import weka.core.jvm as jvm

from classes.config import dirconfig
from classes.Dictionary import Dictionary
from classes.UFSACO import UFSACO
from classes.ClassifierDecisionTreeJ48 import ClassifierDecisionTreeJ48
from classes.ClassifierNaiveBayes import ClassifierNaiveBayes


def main(configFileName):
    """
    UFSACO algorithm execution.
    In order to perform the feature selection, a configuration file must be provided:
    -f <file_name>: file name with JSON extension inside the "ufsacoconf" folder. This folder
                    must be configured inside /acofeatures/classes/config/dirconfig.py file
    :return:
    """
    # Load configuration from specified file
    configFile = dirconfig.ufsacoConfigPath + configFileName + '.json'

    if os.path.exists(configFile):
        with open(configFile, 'r') as configFile:
            configuration = json.loads(configFile.read())
            configFile.close()

        configOptions = ['numberAnts', 'numberFeatures', 'topFeatures']
        configExtraOptions = ['numberCycles', 'decayRate', 'beta', 'initialPheromone', 'exploreExploitCoeff']

        # Verify required values from configuration are correct, otherwise terminate process.
        for optionValue in configOptions:
            if optionValue not in configuration:
                print 'Value for ' + optionValue + ' is missing in configuration. Execution aborted.'
                exit()

        # Add extra configuration
        optionalConfig = {}
        for extraOption in configExtraOptions:
            if extraOption in configuration:
                optionalConfig[extraOption] = configuration[extraOption]
            else:
                optionalConfig[extraOption] = None

        # Get top feature number to use in classification process for UFSACO, Information Gain and Gain Ratio
        topFeatures = configuration['topFeatures']

        # Initialize UFSACO algorithm
        aco = UFSACO(
            numberAnts=configuration['numberAnts'],
            numberFeatures=configuration['numberFeatures'],
            dictionaryName='training',
            dictionaryFolderHier='',
            numberCycles=optionalConfig['numberCycles'],
            decayRate=optionalConfig['decayRate'],
            beta=optionalConfig['beta'],
            initialPheromone=optionalConfig['initialPheromone'],
            exploreExploitCoeff=optionalConfig['exploreExploitCoeff']
        )

        # Perform feature selection using UFSACO
        aco.searchSubset()

        """
        Evaluation: performance of the following classifiers:
        * Decision trees using J48 algorithm
        * Naive Bayes

        To evaluate performance, use the following feature selection values:
        * UFSACO
        * Information Gain
        * Gain Ratio
        """
        # Get dictionary from ACO
        trainingDict = aco.dictionary

        # Store feature list
        featureList = {}

        # Get top-N feature selection subset from UFSACO
        featureList['ufsaco'] = aco.getFeatureResults(topNumber=topFeatures)

        # Get top-N feature selection subset from Information Gain
        featureList['info_gain'] = trainingDict.getInformationGainTopFeatures(topNumber=topFeatures,
                                                                              onlyTokens=True)

        # Get top-N feature selection subset from Gain Ratio
        featureList['gain_ratio'] = trainingDict.getGainRatioTopFeatures(topNumber=topFeatures, onlyTokens=True)

        # Load test dictionary
        testDictionary = Dictionary(dictionaryName='test', folderHierarchy='')
        testDictionary.loadFromDisk()

        try:
            # Start JVM
            jvm.start(max_heap_size='1g')

            for featureType in featureList:
                # Create ARFF file for training
                trainingArffFileName = trainingDict.dictionaryName + '-' + featureType
                trainingDict.createArffFile(arffFileName=trainingArffFileName, tokenList=featureList[featureType])

                # Create ARFF file for testing
                testArffFileName = testDictionary.dictionaryName + '-' + featureType
                testDictionary.createArffFile(arffFileName=testArffFileName, tokenList=featureList[featureType])

                # After creating ARFF files, test using classification
                # Create Decision Tree classifier instance
                j48classifier = ClassifierDecisionTreeJ48(arffFileName=trainingArffFileName)

                # Generate unpruned tree
                j48classifier.setUnprunedTree(True)

                # Create Naive Bayes classifier instance
                nbClassifier = ClassifierNaiveBayes(arffFileName=trainingArffFileName)

                # Build J48 classifier
                j48ClassifierBuilt = j48classifier.build()

                # Evaluate J48 classifier using test data
                if j48ClassifierBuilt is True:
                    j48EvaluationSuccess = j48classifier.testDataEvaluate(testDataArffFileName=testArffFileName)

                    # Show evaluation results
                    if j48EvaluationSuccess is True:
                        print j48classifier.evaluationResults['percent_correct']

                # Build Naive Bayes classifier
                nbClassifierBuilt = nbClassifier.build()

                # Evaluate Naive Bayes classifier using test data
                if nbClassifierBuilt is True:
                    nbEvaluationSuccess = nbClassifier.testDataEvaluate(testDataArffFileName=testArffFileName)

                    # Show evaluation results
                    if nbEvaluationSuccess is True:
                        print nbClassifier.evaluationResults['percent_correct']

        finally:
            if jvm.started is True:
                jvm.stop()  # Stop JVM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute UFSACO algorithm. The algorithm is compared with Information Gain and Gain Ratio feature selections using two different classifiers: Decision Trees and Naive Bayes.")

    parser.add_argument("-f",
                        metavar='FILE_NAME_CONFIGURATION',
                        type=str,
                        help="File name to extract configuration for UFSACO algorithm. Files are obtained from the \"ufsacoconf\" folder, configured in dirconfig.py. Do not include JSON extension, it is added automatically.")

    args = parser.parse_args()
    sys.exit(main(configFileName=args.f))
