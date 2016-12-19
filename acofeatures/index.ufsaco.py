import sys
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
    print aco.getFeatureResults(topNumber=10)


if __name__ == '__main__':
    sys.exit(main())
