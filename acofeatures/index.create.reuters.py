# coding=utf-8
import sys
from nltk.corpus import reuters
from classes.Dictionary import Dictionary


def main():
    """
    Process and index Reuters documents divided into categories (training and test)
    :return:
    """
    # Categories to extract documents
    reutersCategories = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

    # Initialize documents for category
    documents = {
        'training': {},
        'test': {}
    }

    # Different categories for dictionary
    dictionaryCategories = {
        'training': set(),
        'test': set()
    }

    # Get IDs for documents in Reuters corpora
    documentItems = reuters.fileids(reutersCategories)

    # Iterate on retrieved documents for the category
    for documentItem in documentItems:
        documentIdent = documentItem.split('/')

        if len(documentIdent) == 2:
            # Get document type (training|tests)
            documentType = documentIdent[0]

            # Get document ID
            docId = documentIdent[1]

            # Get words from document
            words = reuters.words(documentItem)

            # Get document classes
            docClasses = set(reuters.categories(documentItem)).intersection(reutersCategories)

            # Take only the first class associated
            if len(docClasses) > 0:
                docClass = list(docClasses)[0]

            # Store document in training or data for a category
            if (documentType == 'training') or (documentType == 'test'):
                documents[documentType][docId] = {'words': words, 'class': docClass}

                # Add categories to dictionary categories
                dictionaryCategories[documentType].add(docClass)

    # List of document set types with flag to claculate similarities in the dictionary
    setTypes = {
        'training': True,  # Calculate similarities between tokens
        'test': False  # Do not calculate similarities between tokens, it won't be used
    }

    # Store items in dictionaries (training and test)
    for documentSetType in setTypes.keys():
        # Create new dictionary
        dictionary = Dictionary(dictionaryName=documentSetType, folderHierarchy='')
        docsTraining = documents[documentSetType]

        # Process all documents in set
        for docId in docsTraining:
            """
            Store tokens associated to a document and the class.
            To create a dictionary with different data, this method will
            be used to store every feature to a document/item, having
            a class associated
            """
            dictionary.processDocumentTokens(documentId=docId,
                                             tokens=docsTraining[docId]['words'],
                                             documentClass=docsTraining[docId]['class']
                                             )

        # In case there are documents processed, store in disk and calculate similarities if needed
        if dictionary.documentCount > 0:
            dictionary.saveToDisk(calculateSimilarities=setTypes[documentSetType])


if __name__ == '__main__':
    sys.exit(main())
