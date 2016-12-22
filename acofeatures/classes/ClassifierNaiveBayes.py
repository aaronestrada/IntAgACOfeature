from weka.classifiers import Classifier
from ClassifierAbstract import ClassifierAbstract


class ClassifierNaiveBayes(ClassifierAbstract):
    """
    Naive Bayes classifier algorithm in Weka
    """
    def build(self):
        """
        Build J48 classifier using data loaded from ARFF
        :param storeModel: Store model after built
        :return:
        """
        try:
            dataLoaded = self.loadClassifierData()

            if dataLoaded is True:
                # Naive Bayes classificator
                print '[Building Naive Bayes from training]'
                self.classifierInstance = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
                self.classifierInstance.build_classifier(self.classificationData)
                return True
        except:
            return False

        return False
