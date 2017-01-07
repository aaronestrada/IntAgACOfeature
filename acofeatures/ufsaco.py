import os
import sys
import json
import time
import argparse
import weka.core.jvm as jvm

from classes.config import dirconfig
from classes.Dictionary import Dictionary
from classes.UFSACO import UFSACO
from classes.ClassifierDecisionTreeJ48 import ClassifierDecisionTreeJ48
from classes.ClassifierNaiveBayes import ClassifierNaiveBayes


def main(configFileName, outputFilePath):
    """
    UFSACO algorithm execution.
    In order to perform the feature selection, a configuration file must be provided:
    -f <file_name>: file name with JSON extension inside the "ufsacoconf" folder. This folder
                    must be configured inside /acofeatures/classes/config/dirconfig.py file

    -o <file_path>: file to store results for evaluation. If not defined, show results on screen.
    :return:
    """
    """
    TASK 1: Load configuration from specified file
    """
    configFile = dirconfig.ufsacoConfigPath + configFileName + '.json'

    if os.path.exists(configFile):
        with open(configFile, 'r') as configFile:
            configuration = json.loads(configFile.read())
            configFile.close()

        # List of mandatory configuration
        configOptions = ['numberAnts', 'numberFeatures', 'topFeatures']

        # List of optional configuration
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

        """
        TASK 2: Perform UFSACO algorithm and evaluate results using two different
        classification models using Weka
        """
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

        # Start time previous to search
        searchStartTime = time.time()

        # Perform feature selection using UFSACO
        aco.searchSubset()

        # Ending time of searching
        executionTime = round(time.time() - searchStartTime, 4)

        """
        Evaluation: performance of the following classifiers:
        * Decision trees using J48 algorithm
        * Naive Bayes

        To evaluate performance, use the following feature selection values:
        * UFSACO
        * Information Gain
        * Gain Ratio
        """
        # Free similiarity matrix for ACO dictionary
        aco.dictionary.freeSimilarities()

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
            # Store classification results
            classificationResult = {}

            # Start JVM. Configure JAVA maximum memory heap as desired
            jvm.start(max_heap_size='2g')

            for featureType in featureList:
                # Store classification results for each feature type on each classification model
                classificationResult[featureType] = {}

                # Create ARFF file for training
                trainingArffFileName = configFileName + '-' + trainingDict.dictionaryName + '-' + featureType
                trainingDict.createArffFile(arffFileName=trainingArffFileName, tokenList=featureList[featureType])

                # Create ARFF file for testing
                testArffFileName = configFileName + '-' + testDictionary.dictionaryName + '-' + featureType
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
                        classificationResult[featureType]['j48'] = j48classifier.evaluationResults

                        # Build Naive Bayes classifier
                nbClassifierBuilt = nbClassifier.build()

                # Evaluate Naive Bayes classifier using test data
                if nbClassifierBuilt is True:
                    nbEvaluationSuccess = nbClassifier.testDataEvaluate(testDataArffFileName=testArffFileName)

                    # Show evaluation results
                    if nbEvaluationSuccess is True:
                        classificationResult[featureType]['naive_bayes'] = nbClassifier.evaluationResults
        finally:
            if jvm.started is True:
                jvm.stop()  # Stop JVM

        """
        TASK 3: Output results of evaluation in file or screen
        """
        # Titles for each feature selection
        typeText = {
            'ufsaco': 'Unsupervised Feature Selection using ACO',
            'info_gain': 'Information gain',
            'gain_ratio': 'Gain ratio'
        }

        # Titles for each classification used
        classificationText = {
            'j48': 'Decision tree (J48)',
            'naive_bayes': 'Naive Bayes'
        }

        outputInFile = False
        if outputFilePath is not None:
            outputInFile = True

        outputStr = '[UFSACO execution time: ' + str(executionTime) + ' seconds]\n\n'
        for featureType in classificationResult:
            outputStr += '-------------------' + ('-' * len(typeText[featureType])) + '\n'
            outputStr += 'Feature selection: ' + typeText[featureType] + '\n'
            outputStr += '-------------------' + ('-' * len(typeText[featureType])) + '\n'

            # Display selected features
            enumeratedFeatures = []
            for num, feature in enumerate(featureList[featureType], start=1):
                enumeratedFeatures.append('[#' + str(num) + '] ' + feature)
            outputStr += 'Selected features: ' + ' '.join(enumeratedFeatures) + '\n\n'

            featureTypeResults = classificationResult[featureType]
            for resultItem in featureTypeResults:
                """
                Display results for each classification using different features. For this example,
                only classification correctness and mean absolute error are displayed. However, it is
                possible to extract any type of metric generated by Weka.
                """
                outputStr += classificationText[resultItem] + ': classification results' + '\n'
                outputStr += '* Correct classification: ' + str(
                    round(featureTypeResults[resultItem]['percent_correct'], 4)) + '%\n'
                outputStr += '* Mean absolute error: ' + str(
                    round(featureTypeResults[resultItem]['mean_absolute_error'], 4)) + '\n\n'

        # Display visited features on each iteration of UFSACO
        outputStr += '--------------------------------------\n'
        outputStr += 'UFSACO: visited features on iterations\n'
        outputStr += '--------------------------------------\n'

        for iteration in aco.featureCounterIteration:
            featureCounter = aco.featureCounterIteration[iteration]

            # Sort feature counter (descendant order by counter value)
            sortedFeatureCounter = sorted(featureCounter, key=featureCounter.__getitem__, reverse=True)

            # Concatenate string as [#num_feature] feature_value (counter_value)
            enumeratedFeatures = []
            for num, feature in enumerate(sortedFeatureCounter, start=1):
                enumeratedFeatures.append(
                    '[#' + str(num) + '] ' + feature + ' (' + str(featureCounter[feature]) + ')'
                )
            outputStr += 'Iteration ' + str(iteration + 1) + ': ' + ' '.join(enumeratedFeatures) + '\n\n'

        # Verify if output is in screen or file
        if outputInFile is False:
            print outputStr
        else:
            # Try to open output file, otherwise show in screen
            try:
                with open(outputFilePath, 'w') as outputFile:
                    outputFile.write(outputStr)
                    outputFile.close()
            except:
                print outputStr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Execute UFSACO algorithm. The algorithm is compared with Information Gain and Gain Ratio feature selections using two different classifiers: Decision Trees and Naive Bayes.")

    # File configuration argument definition
    parser.add_argument("-f",
                        metavar='FILE_NAME_CONFIGURATION',
                        type=str,
                        help="File name to extract configuration for UFSACO algorithm. Files are obtained from the \"ufsacoconf\" folder, configured in dirconfig.py. Do not include JSON extension, it is added automatically.")

    # Results output file argument definition
    parser.add_argument("-o",
                        metavar='RESULTS_OUTPUT_FILE_NAME',
                        type=str,
                        default=None,
                        help="File name to save results.")

    args = parser.parse_args()
    sys.exit(main(configFileName=args.f, outputFilePath=args.o))
