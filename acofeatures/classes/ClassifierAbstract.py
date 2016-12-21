import os
from config import dirconfig
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random


class ClassifierAbstract:
    """
    Classifier abstract class to use with Weka Wrapper
    """

    def __init__(self, arffFileName):
        """
        Class constructor. Set file to load data for classifier
        :param arffFileName:  File name to load data
        """
        # Path for ARFF file
        self.arffFileFullPath = dirconfig.arffPath + arffFileName + '.arff'

        # Classes specification
        self.classList = []
        self.classCounter = 0

        # Instances and attributes from data
        self.numInstances = 0
        self.numAttributes = 0
        self.numLeaves = 0
        self.treeSize = 0
        self.classIndex = None

        # Training data for classification
        self.classificationData = None

        # Classifier instance
        self.classifierInstance = None

        # Classifier options
        self.dtOptions = []

        # Evaluation results
        self.evaluationResults = {}

        # Default number of K folds for cross validation
        self.evaluationNumFolds = 10

    def loadData(self):
        """
        Load data from ARFF file
        :return:
        """
        if os.path.exists(self.arffFileFullPath):
            try:
                arffLoader = Loader(classname="weka.core.converters.ArffLoader")

                # Step 1: load classification training data (from ARFF file)
                print '[Loading ARFF data]'
                self.classificationData = arffLoader.load_file(self.arffFileFullPath)

                # set last attribute as class
                self.classificationData.class_is_last()

                # Number of found classes
                self.classCounter = self.classificationData.class_attribute.num_values

                # List of classes
                self.classList = self.classificationData.class_attribute.values

                # Number of instances
                self.numInstances = self.classificationData.num_instances

                # Number of attributes
                self.numAttributes = self.classificationData.num_attributes

                # Class index inside the data
                self.classIndex = self.classificationData.class_index
                return True
            except:
                return False
        return False

    # Block: Evaluation
    def setCrossValidationKFolds(self, foldsNum):
        """
        Set evaluation folds number (for cross-validation)
        :param foldsNum: Number of folds
        :return:
        """
        self.evaluationNumFolds = foldsNum

    def crossEvaluate(self):
        """
        Evaluate classifier using cross-validation using K folds
        :return:
        """
        if self.classifierInstance is not None:
            try:
                print '[Cross-validate data]'
                evaluatorInstance = Evaluation(self.classificationData)
                evaluatorInstance.crossvalidate_model(self.classifierInstance, self.classificationData,
                                                      self.evaluationNumFolds, Random(1))

                self.evaluationResults = {
                    'num_correct': evaluatorInstance.correct,
                    'percent_correct': evaluatorInstance.percent_correct,
                    'num_incorrect': evaluatorInstance.incorrect,
                    'percent_incorrect': evaluatorInstance.percent_incorrect,
                    'kappa': evaluatorInstance.kappa,
                    'mean_absolute_error': evaluatorInstance.mean_absolute_error,
                    'root_mean_prior_squared_error': evaluatorInstance.root_mean_prior_squared_error,
                    'relative_absolute_error': evaluatorInstance.relative_absolute_error,
                    'root_relative_squared_error': evaluatorInstance.root_relative_squared_error,
                    'num_instances': evaluatorInstance.num_instances,
                    'confussion_matrix': evaluatorInstance.confusion_matrix.tolist()
                }

                # Evaluation results for each class
                classStats = {}
                for classIndex in range(0, self.classCounter):
                    classStats[self.classList[classIndex]] = {
                        'tp': evaluatorInstance.num_true_positives(classIndex),
                        'tn': evaluatorInstance.num_true_negatives(classIndex),
                        'fp': evaluatorInstance.num_false_positives(classIndex),
                        'fn': evaluatorInstance.num_false_negatives(classIndex),
                        'tp_rate': evaluatorInstance.true_positive_rate(classIndex),
                        'tn_rate': evaluatorInstance.true_negative_rate(classIndex),
                        'fp_rate': evaluatorInstance.false_positive_rate(classIndex),
                        'fn_rate': evaluatorInstance.false_negative_rate(classIndex),
                        'precision': evaluatorInstance.precision(classIndex),
                        'recall': evaluatorInstance.recall(classIndex),
                        'f-measure': evaluatorInstance.f_measure(classIndex),
                        'matthews_correlation_coefficient': evaluatorInstance.matthews_correlation_coefficient(
                            classIndex),
                        'area_under_roc': evaluatorInstance.area_under_roc(classIndex),
                        'area_under_prc': evaluatorInstance.area_under_prc(classIndex)
                    }

                self.evaluationResults['classStats'] = classStats
                return True
            except:
                return False
        return False
