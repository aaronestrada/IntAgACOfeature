Unsupervised Feature Selection using Ant Colony Optimization (UFSACO)
===

Simulation of an Unsupervised Feature Selection using Ant Colony Optimization (UFSACO) algorithm. System is implemented in Python 2.7.11.

Details on the algorithm can be found in the following [link](https://www.researchgate.net/publication/261371258).

# 1. Pre-requisites

## Dependencies

The following dependencies must be installed to execute the system:

* nltk ([View documentation](http://www.nltk.org/))
* python-weka-wrapper ([View documentation](http://pythonhosted.org/python-weka-wrapper/index.html))

Installing process for each dependency is detailed on each link.

## Folder configuration

The first step corresponds to configure directories for indexes and ARFF files (used in Weka). Create the file dirconfig.py in /acofeatures/classes/config/ and set the following directories.

```
# Storage of ARFF files generated for classification using Weka
arffPath = '<path_to_project>/arff/'

# Storage of dictionaries (training and test)
dictionaryPath = '<path_to_project>/dictionaries/'

# Storage of user-defined configuration for UFSACO
ufsacoConfigPath = '<path_to_project>/ufsacoconf/'
```
Path <path_to_project> is suggested to be inside the system; however you can set the path anywhere as long as the directory has read/write permissions.

## Dataset

NLTK provides many datasets for text categorization. The following system uses Reuters-21578 corpus on 10 categories. Even when documents belong to many categories, for testing purposes each document will be assigned to only one label. 

## Indexing Reuters-21578 corpus

The second step is constructing an inverted index for the Reuters-21578 corpus. To proceed, run the following command:
```
$ python acofeatures/index.create.reuters.py
```

Two dictionaries will be constructed in path /dictionaries:
* training: contains all the documents to construct the classifiers
* test: documents that will be tested using the constructed classifiers

A dictionary generates the following files:

```
index.documents.json        List of documents with its corresponding class and features
index.gainratio.json        Gain ratio calculations for each feature
index.infogain.json         Information gain calculations for each feature
index.postingdocs.json      Inverted index with features and occurrence on documents
index.postings.json         List of features from the corpus
index.similarities.json     Similarity calculation between each feature (the largest file). 
                            Only used in training index.
stats.idf.json              Inverse document frequency calculations for documents
stats.index.json            General information about index
stats.tf.json               Term frequency calculations for each feature
stats.tfidf.json            TF-IDF calculations for features in documents
```

N.B. Indexing process takes a while to execute, specially for the similarity calculations between features.

# 2. Running UFSACO algorithm
The following command is used to run the algorithm:

```
$ python acofeatures/ufsaco.py -f <file_name_no_extension_included> [-o <output_file_name>]
```

* -f <file_name_no_extension_included> (Mandatory) - Defines the configuration file to run the algorithm.  

* -o <output_file_name> (Optional) - Saves the evaluation for the algorithm on the specified output file.

### Configuration files for algorithm
Configuration files are stored in the /ufsacoconf folder. Folder settings are indicated in /acofeatures/classes/config/dirconfig.py file.

Each configuration file accepts the following parameters (view ufsacoconf/conf.example.json file for reference):
* _numberAnts_: Number of ant agents working as a colony (mandatory).
* _numberFeatures_: Number of features the ants must select in each iteration (mandatory).
* _numberCycles_: Number of iterations the ants will work. In each iteration, an ant will select _numberFeatures_ features (default value: 50)
* _topFeatures_: Number of features to use in the classification task. Prior to build classification models, an ARFF file is constructed using only the number of features. This number will be the same for UFSACO, Information Gain and Gain Ratio feature selection (mandatory).
* _decayRate_: Pheromone decay rate value (default value: 0.2)
* _beta_: Beta value for algorithm (default value: 1)
* _initialPheromone_: Initial pheromone value for all the features (default value: 0.2)
* _exploreExploitCoeff_: Exploration / exploitation coefficient, used to decide the selection of the next feature (default value: 0.7)

Example of configuration file for running algorithm:
```
{
  "numberAnts": 100,
  "numberFeatures": 10,
  "numberCycles": 50,
  "topFeatures": 10,
  "decayRate": 0.2,
  "beta": 1,
  "initialPheromone": 0.2,
  "exploreExploitCoeff": 0.7
}
```

For instance, to run the example configuration file, use the following command:
```
$ python acofeatures/ufsaco.py -f conf.example
```