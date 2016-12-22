from weka.classifiers import Classifier
from ClassifierAbstract import ClassifierAbstract


class ClassifierDecisionTreeJ48(ClassifierAbstract):
    """
    Decision tree using J48 algorithm in Weka
    """

    def __init__(self, arffFileName, confidenceValue=0.25):
        """
        Class constructor (overridden)
        :param arffFileName: ARFF file name
        :param confidenceValue: Confidence value for classifier
        """
        ClassifierAbstract.__init__(self, arffFileName)

        # Store confidence value
        if 0 <= confidenceValue <= 1:
            self.confidenceValue = confidenceValue
        else:
            # Set default confidence value
            self.confidencevalue = 0.25

        self.unpruned = False

    def setUnprunedTree(self, unpruned):
        """
        Set unpruned tree option
        :param unpruned: If tree result is unpruned or not (TRUE or FALSE)
        :return:
        """
        self.unpruned = unpruned

    def build(self):
        """
        Build J48 classifier using data loaded from ARFF
        :param storeModel: Store model after built
        :return:
        """
        try:
            dataLoaded = self.loadClassifierData()

            if dataLoaded is True:
                # Decision tree options
                if self.unpruned is True:
                    self.dtOptions = ['-U']
                else:
                    self.dtOptions = ['-C', str(self.confidenceValue)]

                # Decision tree classificator
                print '[Building J48 DT from training]'
                self.classifierInstance = Classifier(classname="weka.classifiers.trees.J48", options=self.dtOptions)
                self.classifierInstance.build_classifier(self.classificationData)
                return True
        except:
            return False

        return False
