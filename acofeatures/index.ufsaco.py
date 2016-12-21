import sys
import weka.core.jvm as jvm

from classes.Dictionary import Dictionary
from classes.UFSACO import UFSACO
from classes.ClassifierDecisionTreeJ48 import ClassifierDecisionTreeJ48


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
    # Create Decision Tree instance
    j48classifier = ClassifierDecisionTreeJ48(arffFileName=trainingArffFileName, confidenceValue=0.25)

    try:
        # Start JVM
        jvm.start(max_heap_size='1g')

        # Build classifier
        classifierBuilt = j48classifier.build()

        # Evaluate classifier using test data
        if classifierBuilt is True:
            evaluationSuccess = j48classifier.testDataEvaluate(testDataArffFileName=testArffFileName)

            # Show evaluation results
            if evaluationSuccess is True:
                print j48classifier.evaluationResults

    finally:
        if jvm.started is True:
            jvm.stop()  # Stop JVM


if __name__ == '__main__':
    sys.exit(main())
