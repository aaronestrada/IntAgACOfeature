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

    def build(self):
        """
        Build J48 classifier using data loaded from ARFF
        :param storeModel: Store model after built
        :return:
        """
        try:
            dataLoaded = self.loadData()

            if dataLoaded is True:
                # Decision tree options
                self.dtOptions = ['-C', str(self.confidenceValue)]

                # Decision tree classificator
                print '[Building from training]'
                self.classifierInstance = Classifier(classname="weka.classifiers.trees.J48", options=self.dtOptions)
                self.classifierInstance.build_classifier(self.classificationData)
                self.numLeaves = self.classifierInstance.jwrapper.measureNumLeaves()
                self.treeSize = self.classifierInstance.jwrapper.measureTreeSize()

                return True
        except:
            return False

        return False