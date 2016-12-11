# coding=utf-8
import sys
from nltk.corpus import reuters
from classes.Dictionary import Dictionary


def main():
    """
    Process and index Reuters documents divided into categories (training and test)
    :return:
    """

    # List of documents for category
    categoryDocuments = {}

    # Get Reuters categories
    # print reuters.categories()

    # Categories to extract documents
    reutersCategories = ['acq', 'corn', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']

    for category in reutersCategories:
        # Initialize documents for category
        categoryDocuments[category] = {
            'training': {},
            'test': {}
        }

        # Get IDs for documents in Reuters corpora
        categoryDocumentItems = reuters.fileids(category)

        # Iterate on retrieved documents for the category
        for documentItem in categoryDocumentItems:
            documentIdent = documentItem.split('/')

            if len(documentIdent) == 2:
                # Get document type (training|tests)
                documentType = documentIdent[0]

                # Get document ID
                docId = documentIdent[1]

                # Get words from document
                words = reuters.words(documentItem)

                # Raw document
                # print(reuters.raw(documentItem));

                # Store document in training or data for a category
                if (documentType == 'training') or (documentType == 'test'):
                    categoryDocuments[category][documentType][docId] = words

    # List of document set types
    setTypes = ['training', 'test']

    # Store items in dictionaries (training and test)
    for documentSetType in setTypes:
        for category in categoryDocuments:
            # Create new dictionary
            dictionary = Dictionary(dictionaryName=category, folderHierarchy=documentSetType)
            docsTraining = categoryDocuments[category][documentSetType]

            # Process all documents in set
            for docId in docsTraining:
                dictionary.processDocumentTokens(docId, docsTraining[docId])

            # In case there are documents processed, store in disk
            if dictionary.documentCount > 0:
                dictionary.saveToDisk()


if __name__ == '__main__':
    sys.exit(main())
