Unsupervised Feature Selection using Ant Colony Optimization (UFSACO)
===

Simulation of Unsupervised feature selection using Ant Colony Optimization (UFSACO) algorithm. System is implemented in Python 2.7.11.

Details on the algorithm can be found in the following link: https://www.researchgate.net/publication/261371258

Dependencies
---
The following dependencies must be installed to execute the system:

* nltk (http://www.nltk.org/)
* python-weka-wrapper (http://pythonhosted.org/python-weka-wrapper/index.html) 

Installing process for each dependency is detailed on each link.
 
Dataset
---
NLTK provides many datasets for text categorization. The following system uses Reuters-21578 corpus on 10 categories. Even when documents belong to many categories, for testing purposes each document will be assigned to only one label. 

Configuration
---
The first step corresponds to configure directories for indexes and ARFF files (used in Weka). Create the file dirconfig.py in /acofeatures/classes/config/ and set the following directories.

```
arffPath = '<path_to_project>/arff/'
dictionaryPath = '<path_to_project>/dictionaries/'
```
Path <path_to_project> is suggested to be inside the system; however you can set the path anywhere as long as the directory has read/write permissions.

Indexing process
---
The second step is constructing an inverted index for the Reuters-21578 corpus. To proceed, run the following command:
```
$ python acofeatures/index.create.reuters.py
```

Two dictionaries will be constructed in path /dictionaries:
* training: contains all the documents to construct the classifiers
* test: documents that will be tested using the constructed classifiers



