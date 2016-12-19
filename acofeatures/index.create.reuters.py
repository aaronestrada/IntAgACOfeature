# coding=utf-8
import sys
from nltk.corpus import reuters
from classes.Dictionary import Dictionary


def main():
    """
    Process and index Reuters documents divided into categories (training and test)
    :return:
    """

    # List of documents
    documents = {}

    # Get Reuters categories
    # print reuters.categories()

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
                dictionaryCategories[documentType] = dictionaryCategories[documentType].union(docClass)

    # List of document set types
    setTypes = ['training', 'test']

    # Store items in dictionaries (training and test)
    for documentSetType in setTypes:
        # Create new dictionary
        dictionary = Dictionary(dictionaryName=documentSetType, folderHierarchy='')
        docsTraining = documents[documentSetType]

        # Save found categories in dictionary
        dictionary.setCategories(list(dictionaryCategories[documentSetType]))

        # Process all documents in set
        for docId in docsTraining:
            dictionary.processDocumentTokens(docId, docsTraining[docId]['words'], docsTraining[docId]['class'])

        # In case there are documents processed, store in disk
        if dictionary.documentCount > 0:
            dictionary.saveToDisk()


if __name__ == '__main__':
    sys.exit(main())
