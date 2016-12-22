import sys
import weka.core.jvm as jvm

from classes.Dictionary import Dictionary
from classes.UFSACO import UFSACO
from classes.ClassifierDecisionTreeJ48 import ClassifierDecisionTreeJ48
from classes.ClassifierNaiveBayes import ClassifierNaiveBayes


def main():
    """
    UFSACO algorithm testing
    :return:
    """
    aco = UFSACO(
        numberAnts=10,
        numberFeatures=10,
        numberCycles=10,
        dictionaryName='training',
        dictionaryFolderHier=''
    )

    # Perform feature selection
    aco.searchSubset()

    # Get top-10 feature selection subset
    acoFeatureList = aco.getFeatureResults(topNumber=10)

    # Create ARFF file for training
    trainingDict = aco.dictionary  # Get used dictionary on ACO

    trainingArffFileName = trainingDict.dictionaryName + '-aco'
    trainingDict.createArffFile(arffFileName=trainingArffFileName, tokenList=acoFeatureList)

    # Load test dictionary and create ARFF file for testing
    testDictionary = Dictionary(dictionaryName='test', folderHierarchy='')
    testDictionary.loadFromDisk()
    testArffFileName = testDictionary.dictionaryName + '-aco'
    testDictionary.createArffFile(arffFileName=testArffFileName, tokenList=acoFeatureList)

    # After creating ARFF files, test using classification
    # Create Decision Tree classifier instance
    j48classifier = ClassifierDecisionTreeJ48(arffFileName=trainingArffFileName)

    # Generate unpruned tree
    j48classifier.setUnprunedTree(True)

    # Create Naive Bayes classifier instance
    nbClassifier = ClassifierNaiveBayes(arffFileName=trainingArffFileName)

    try:
        # Start JVM
        jvm.start(max_heap_size='1g')

        # Build classifier
        j48ClassifierBuilt = j48classifier.build()

        # Evaluate J48 classifier using test data
        if j48ClassifierBuilt is True:
            j48EvaluationSuccess = j48classifier.testDataEvaluate(testDataArffFileName=testArffFileName)

            # Show evaluation results
            if j48EvaluationSuccess is True:
                print j48classifier.evaluationResults

        nbClassifierBuilt = nbClassifier.build()

        # Evaluate Naive Bayes classifier using test data
        if nbClassifierBuilt is True:
            nbEvaluationSuccess = nbClassifier.testDataEvaluate(testDataArffFileName=testArffFileName)

            # Show evaluation results
            if nbEvaluationSuccess is True:
                print nbClassifier.evaluationResults


    finally:
        if jvm.started is True:
            jvm.stop()  # Stop JVM


if __name__ == '__main__':
    sys.exit(main())
