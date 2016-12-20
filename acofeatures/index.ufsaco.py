import sys
from classes.Dictionary import Dictionary
from classes.UFSACO import UFSACO


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
    trainingDict = aco.dictionary
    trainingDict.createArffFile(arffFileName=trainingDict.dictionaryName + '-aco', tokenList=acoFeatureList)

    # Load test dictionary and create ARFF file for testing
    testDictionary = Dictionary(dictionaryName='test', folderHierarchy='')
    testDictionary.loadFromDisk()
    testDictionary.createArffFile(arffFileName=testDictionary.dictionaryName + '-aco', tokenList=acoFeatureList)

if __name__ == '__main__':
    sys.exit(main())
