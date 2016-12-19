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

        # Similarity matrix between tokens
        self.similarityMatrix = {}

        # Count of documents in index
        self.documentCount = 0

        # Count of terms in index
        self.termCount = 0

        self.dictionaryName = dictionaryName

        if folderHierarchy != '':
            folderHierarchy += '/'  # Add slash for directory management

        # Store path for dictionary
        self.dictionaryPath = dirconfig.dictionaryPath + folderHierarchy + dictionaryName + '/'

    def createDictionaryPath(self):
        """
        Create dictionary path folder in case it does not exist
        :return:
        """
        if not os.path.exists(self.dictionaryPath):
            os.makedirs(self.dictionaryPath)

    def processDocumentTokens(self, documentId, tokens, documentClass):
        """
        Process all the tokens of a document
        :param documentId: Document ID
        :param tokens: Tokens for the document
        :param documentClass: Class for the document
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

            """
            Store document ID in list and number of different tokens.
            For each document, store the following values:
            * l: Number of different tokens in the document
            * t: Each of the tokens and its counter
            * c: Class for the document
            """
            self.documents[documentId] = {
                'l': tokenCounterLen,
                't': tokenCounter,
                'c': documentClass
            }

            # Incremet document counter
            self.documentCount += 1

    def setCategories(self, categoryList):
        """
        Store list of categories in dictionary
        :param categoryList: List of categories
        :return:
        """
        self.categories = categoryList

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

    def calculateCosineSim(self, documentList, documentsToken1, documentsToken2):
        """
        Calculate cosine similarity between
        :param documentList: List of shared documents between token 1 and token 2
        :param documentsToken1: List of documents for token 1
        :param documentsToken2: List of documents for token 2
        :return:
        """
        totalNum = 0
        totalTokenFrom = 0
        totalTokenTo = 0

        # For each document in shared list, calculate similarity
        for docId in documentList:
            a = documentsToken1[docId]
            b = documentsToken2[docId]

            totalNum += (a * b)
            totalTokenFrom += (a ** 2)
            totalTokenTo += (b ** 2)

        # Verify denominator part is not zero to perform calculation
        totalDen = totalTokenFrom * totalTokenTo

        if totalDen > 0:
            return round(float(totalNum) / math.sqrt(totalDen), 4)

        return 0

    def calculateAllSimilarities(self):
        """
        Calculate similarities between tokens. This method stores values with no redundancy, i.e.
        for similarities values sim(a,b) = sim(b,a) is only stored once.

        Also, for similarities equal to zero, value is not stored. At the end of the method,
        resulting file is stored.
        :return:
        """
        print 'Calculating similarities between tokens...'
        similarityMatrix = {}

        # Go token by token to calculate the similarities between the rest of tokens
        for tokenFrom in self.postings:
            documentFrom = self.postingDocuments[tokenFrom]
            documentFromItems = set(documentFrom.keys())

            # Get rest of tokens to calculate values
            for tokenTo in self.postings:
                if not (tokenFrom == tokenTo
                        and tokenTo in similarityMatrix
                        and tokenFrom in similarityMatrix[tokenTo]
                        ):
                    documentTo = self.postingDocuments[tokenTo]
                    similarDocuments = documentFromItems.intersection(documentTo.keys())

                    # If there are similar documents, calculate cosine similarity
                    if len(similarDocuments) > 0:
                        cosineSim = self.calculateCosineSim(documentList=similarDocuments,
                                                            documentsToken1=documentFrom,
                                                            documentsToken2=documentTo)

                        if cosineSim > 0:
                            if tokenFrom not in similarityMatrix:
                                similarityMatrix[tokenFrom] = {}

                            similarityMatrix[tokenFrom][tokenTo] = cosineSim

        # Store similarities in class
        self.similarityMatrix = similarityMatrix

        # Store calculation results in file
        with open(self.dictionaryPath + fileconfig.similarityFileName, 'w') as similarityFile:
            similarityFile.write(json.dumps(similarityMatrix))
            similarityFile.close()

    def loadSimilarities(self):
        """
        Load similarity values calculations
        :return:
        """
        similarityMatrixFilePath = self.dictionaryPath + fileconfig.similarityFileName
        if os.path.exists(similarityMatrixFilePath):
            with open(similarityMatrixFilePath, 'r') as similarityFile:
                self.similarityMatrix = json.loads(similarityFile.read())
                similarityFile.close()

    def calculateTokenSimilarity(self, token, tokenList):
        """
        Calculate similarities between tokens and a list of tokens
        :param token: Token to check similarity
        :param tokenList: List of tokens to compare
        :return: Most similar token
        """
        similarityVector = {}

        # Verify if token exists in dictionary
        if token in self.postingDocuments:
            # Get list of documents and total of repetitions of the token
            documentFrom = self.postingDocuments[token]

            # Get document list as set
            documentFromItems = set(documentFrom.keys())

            for tokenTo in tokenList:
                # Get list of documents to other tokens
                documentTo = self.postingDocuments[tokenTo]
                similarDocuments = documentFromItems.intersection(documentTo.keys())

                # If there are similar documents, calculate cosine similarity
                if len(similarDocuments) > 0:
                    cosineSim = self.calculateCosineSim(documentList=similarDocuments,
                                                        documentsToken1=documentFrom,
                                                        documentsToken2=documentTo)

                    if cosineSim > 0:
                        similarityVector[tokenTo] = cosineSim

            orderedSimilarity = sorted(similarityVector, key=similarityVector.__getitem__, reverse=True)
            return orderedSimilarity[0]

        # In case token does not exist, return none
        return None

    def getSimilarity(self, token1, token2):
        """
        Get similarity values between two tokens. Works only if similarity values
        have been calculated before
        :param token1: Token 1 to check similarity
        :param token2: Token 2 to check similarity
        :return: Tokens similarity
        """
        if token1 in self.similarityMatrix and token2 in self.similarityMatrix[token1]:
            return self.similarityMatrix[token1][token2]
        elif token2 in self.similarityMatrix and token1 in self.similarityMatrix[token2]:
            return self.similarityMatrix[token2][token1]
        else:
            return 0

    def getTokenSimilarity(self, token, tokenList):
        """
        Get similarities between tokens and a list of tokens. This method
        makes use of the similarity matrix loaded from disk to search
        between similarity values.
        :param token: Token to check similarity
        :param tokenList: List of tokens to compare
        :return: Most similar token
        """
        similarityVector = {}
        for token2 in tokenList:
            simValue = self.getSimilarity(token1=token, token2=token2)

            if simValue > 0:
                similarityVector[token2] = simValue

        if len(similarityVector) > 0:
            orderedSimilarity = sorted(similarityVector, key=similarityVector.__getitem__, reverse=True)
            return orderedSimilarity[0]

        # In case no similarities found, return None
        return None

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
                'documents': self.documentCount,
                'categories': self.categories
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

        # Calculate token similarities
        # self.calculateAllSimilarities()

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
                if 'categories' in indexStats:
                    self.categories = indexStats['categories']
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
