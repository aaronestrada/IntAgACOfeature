import os
import json

import math

from config import dirconfig
from config import fileconfig


class Dictionary:
    def __init__(self, dictionaryName, folderHierarchy=''):
        """
        Dictionary constructor
        :param dictionaryName: Name of dictionary to store in disk
        """
        self.postings = []
        self.documents = {}
        self.postingDocuments = {}

        # TF-IDF calculations storage
        self.tf = {}
        self.tfidf = {}
        self.idf = {}

        # Count of documents in index
        self.documentCount = 0

        # Count of terms in index
        self.termCount = 0

        self.dictionaryName = dictionaryName

        if folderHierarchy != '':
            folderHierarchy += '/'

        self.dictionaryPath = dirconfig.dictionaryPath + folderHierarchy + dictionaryName + '/'

    def createDictionaryPath(self):
        """
        Create dictionary path folder in case it does not exist
        :return:
        """
        if not os.path.exists(self.dictionaryPath):
            os.makedirs(self.dictionaryPath)

    def processDocumentTokens(self, documentId, tokens):
        """
        Process all the tokens of a document
        :param documentId: Document ID
        :param tokens: Tokens for the document
        :return:
        """

        # Count occurrences of each token in the document
        tokenCounter = {}
        for token in tokens:
            # Lowercase token
            token = token.lower()

            # Verify if token is not in list
            if token not in tokenCounter:
                tokenCounter[token] = 0

            tokenCounter[token] += 1

        # Process tokens (words) for a document
        tokenCounterLen = len(tokenCounter)
        if tokenCounterLen > 0:
            for token in tokenCounter:
                self.storeDocumentToken(documentId, token, tokenCounter[token])

            # Store document ID in list and number of different tokens
            self.documents[documentId] = tokenCounterLen

            # Incremet document counter
            self.documentCount += 1

    def storeDocumentToken(self, documentId, token, tokenCount):
        """
        Store token for a document in posting items
        :param documentId: Document ID
        :param token: Token
        :param tokenCount: Count of the token in the document
        :return:
        """
        if token not in self.postingDocuments:
            # Add token to posting documents and posting items
            self.postingDocuments[token] = {}
            self.postings.append(token)

            # Increment term counter
            self.termCount += 1

        # Add counter of token in the document
        self.postingDocuments[token][documentId] = tokenCount

    def calculateTfIdf(self):
        """
        Calculation of Tf-Idf values for token
        TF(t,d) - Log term frequency = 1 + log(<Term frequency in document>)
        IDF(t) - Inverted document frequency for a term = log(<Number of documents> / <Document frequency(term)>)
        :return:
        """
        floatDocumentCount = float(self.documentCount)

        for token in self.postings:
            # Get document list for the token
            tokenDocumentList = self.postingDocuments[token]

            # Step 1: calculate IDF
            tokenIdf = math.log10(floatDocumentCount / float(len(tokenDocumentList)))

            # Step 2: calculate Tf-Idf
            tf = {}
            tfidf = {}

            for documentId in tokenDocumentList:
                tokenDocumentTf = float(1 + math.log10(tokenDocumentList[documentId]))
                tf[documentId] = round(tokenDocumentTf, 4)
                tfidf[documentId] = round(tokenDocumentTf * tokenIdf)

            self.tf[token] = tf
            self.tfidf[token] = tfidf
            self.idf[token] = round(tokenIdf, 4)

    def saveToDisk(self):
        """
        Save index content to disk
        :return:
        """
        # Create folder (if needed)
        self.createDictionaryPath()

        # Dump index stats
        with open(self.dictionaryPath + fileconfig.indexStatsFileName, 'w') as indexStatsFile:
            indexStatsFile.write(json.dumps({
                'terms': self.termCount,
                'documents': self.documentCount
            }))
            indexStatsFile.close()

        # Dump posting documents
        with open(self.dictionaryPath + fileconfig.postingDocsFileName, 'w') as postingDocumentsFile:
            postingDocumentsFile.write(json.dumps(self.postingDocuments))
            postingDocumentsFile.close()

        # Dump document list
        with open(self.dictionaryPath + fileconfig.documentsFileName, 'w') as documentsFile:
            documentsFile.write(json.dumps(self.documents))
            documentsFile.close()

        # Dump postings
        with open(self.dictionaryPath + fileconfig.postingsFileName, 'w') as postingsFile:
            postingsFile.write(json.dumps(self.postings))
            postingsFile.close()

        # Calculate TF-IDF
        self.calculateTfIdf()

        # Dump tf values
        with open(self.dictionaryPath + fileconfig.tfFileName, 'w') as tfFile:
            tfFile.write(json.dumps(self.tf))
            tfFile.close()

        # Dump idf values
        with open(self.dictionaryPath + fileconfig.idfFileName, 'w') as idfFile:
            idfFile.write(json.dumps(self.idf))
            idfFile.close()

        # Dump tfidf values
        with open(self.dictionaryPath + fileconfig.tfidfFileName, 'w') as tfidfFile:
            tfidfFile.write(json.dumps(self.tfidf))
            tfidfFile.close()

    def loadFromDisk(self):
        """
        Load dictionary items from disk to memory
        :return:
        """
        # Get index stats
        indexStatsFilePath = self.dictionaryPath + fileconfig.indexStatsFileName
        if os.path.exists(indexStatsFilePath):
            with open(indexStatsFilePath, 'r') as indexStatsFile:
                indexStats = json.loads(indexStatsFile.read())
                if 'terms' in indexStats:
                    self.termCount = indexStats['terms']
                if 'documents' in indexStats:
                    self.documentCount = indexStats['documents']
                indexStatsFile.close()

        # Get posting documents
        postingDocsFilePath = self.dictionaryPath + fileconfig.postingDocsFileName
        if os.path.exists(postingDocsFilePath):
            with open(postingDocsFilePath, 'r') as postingDocumentsFile:
                self.postingDocuments = json.loads(postingDocumentsFile.read())
                postingDocumentsFile.close()

        # Get document list
        documentFilePath = self.dictionaryPath + fileconfig.documentsFileName
        if os.path.exists(documentFilePath):
            with open(documentFilePath, 'r') as documentsFile:
                self.documents = json.loads(documentsFile.read())
                documentsFile.close()

        # Get postings
        postingsFilePath = self.dictionaryPath + fileconfig.postingsFileName
        if os.path.exists(postingsFilePath):
            with open(postingsFilePath, 'r') as postingsFile:
                self.postings = json.loads(postingsFile.read())
                postingsFile.close()

        # Get tfidf values
        tfidfFilePath = self.dictionaryPath + fileconfig.tfidfFileName
        if os.path.exists(tfidfFilePath):
            with open(tfidfFilePath, 'r') as tfidfFile:
                self.tfidf = json.loads(tfidfFile.read())
                tfidfFile.close()
