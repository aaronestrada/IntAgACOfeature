import os
import json
import math
import re
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

        # Token information gain and gain ratio values
        self.tokenInfoGain = {}
        self.orderedTokenInfoGain = {}

        self.tokenGainRatio = {}
        self.orderedTokenGainRatio = {}

        # Category list
        self.categories = {}

        # Similarity matrix between tokens
        self.similarityMatrix = {}

        # Count of documents in index
        self.documentCount = 0

        # Count of terms in index
        self.termCount = 0

        # Store dictionary name
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

            # Add category and calculate counter
            if documentClass not in self.categories:
                self.categories[documentClass] = 0

            # Increment category counter
            self.categories[documentClass] += 1

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

    def calculateInformationGainAndGainRatio(self):
        # Step 1: Calculate information gain bits for all documents
        informationGainDocs = 0
        for category in self.categories:
            probabilityClass = float(self.categories[category]) / self.documentCount

            if probabilityClass > 0:
                informationGainDocs += (-probabilityClass * math.log(probabilityClass, 2))

        # Step 1: Calculate information gain bits for each token in documents
        documentItems = set(self.documents.keys())  # Keep all documents as set

        for token in self.postings:
            # Information gain of token
            tokenIGValue = 0

            # Step 1.1: Calculate probabilities of token in each class
            categoryCounter = {}
            tokenDocuments = self.postingDocuments[token].keys()

            # Get count of documents containing the token
            tokenDocumentCount = len(tokenDocuments)

            # Get class for all the documents containing the token, store in categoryCounter
            for docId in tokenDocuments:
                documentClass = self.documents[docId]['c']

                if documentClass not in categoryCounter:
                    categoryCounter[documentClass] = 0

                categoryCounter[documentClass] += 1

            # Calculate information gain of each category for documents having the token
            informationGainToken = 0
            for category in categoryCounter:
                probabilityCategoryClass = float(categoryCounter[category]) / tokenDocumentCount

                if probabilityCategoryClass > 0:
                    informationGainToken += (-probabilityCategoryClass * math.log(probabilityCategoryClass, 2))

            # Add calculated value to token information gain
            tokenIGValue += (float(tokenDocumentCount) / self.documentCount) * informationGainToken

            # Step 1.2: Calculate probabilities of no token in each class
            categoryNoTokenCounter = {}
            noTokenDocuments = documentItems.difference(tokenDocuments)

            # Get total of documents not having the token
            tokenNotDocumentCount = self.documentCount - tokenDocumentCount

            for docId in noTokenDocuments:
                documentClass = self.documents[docId]['c']

                if documentClass not in categoryNoTokenCounter:
                    categoryNoTokenCounter[documentClass] = 0

                categoryNoTokenCounter[documentClass] += 1

            # Calculate information gain of each category for documents not having the token
            informationGainNoToken = 0
            for category in categoryNoTokenCounter:
                probabilityCategoryClass = float(categoryNoTokenCounter[category]) / tokenNotDocumentCount

                if probabilityCategoryClass > 0:
                    informationGainNoToken += (-probabilityCategoryClass * math.log(probabilityCategoryClass, 2))

            # Add calculated value to token information gain
            tokenIGValue += (float(tokenNotDocumentCount) / self.documentCount) * informationGainNoToken

            # Calculate split info for token (Entropy for different values in token)
            probabilityToken = float(tokenDocumentCount) / self.documentCount
            probabilityNoToken = float(tokenNotDocumentCount) / self.documentCount

            splitInfoToken = -(probabilityToken * math.log(probabilityToken, 2)) \
                             - (probabilityNoToken * math.log(probabilityNoToken, 2))

            # Calculate token gain
            tokenGain = informationGainDocs - tokenIGValue

            # Store token information gain value
            self.tokenInfoGain[token] = tokenGain
            self.tokenGainRatio[token] = float(tokenGain) / splitInfoToken

    def getTopFeatures(self, topNumber, method, onlyTokens=True):
        """
        Return top features using different statistical feature selection
        :param topNumber: Top number of tokens to retrieve
        :param method: Method to get the features (information_gain or gain_ratio)
        :param onlyTokens: Get only tokens, otherwise, get values and tokens
        :return: List of tokens | tokens and values
        """
        if method == 'information_gain':
            orderedFeatures = self.orderedTokenInfoGain
        elif method == 'gain_ratio':
            orderedFeatures = self.orderedTokenGainRatio
        else:
            return False

        # Verify top number is not bigger than length of features
        if len(orderedFeatures) >= topNumber:
            orderedFeatures = orderedFeatures[0:topNumber]

        # Return only token list
        if onlyTokens is True:
            return orderedFeatures
        else:
            # Return topNumber features
            featureResults = {}
            for token in orderedFeatures:
                featureResults[token] = self.tokenInfoGain[token]

            return featureResults

    def getInformationGainTopFeatures(self, topNumber, onlyTokens=True):
        """
        Get feature relevance using information gain values
        :param topNumber: Number of features to retrieve
        :param onlyTokens: TRUE to extract only features, otherwise get IG value as well
        :return:
        """
        return self.getTopFeatures(topNumber=topNumber, onlyTokens=onlyTokens, method='information_gain')

    def getGainRatioTopFeatures(self, topNumber, onlyTokens=True):
        """
        Get feature relevance using gain ratio values
        :param topNumber: Number of features to retrieve
        :param onlyTokens: TRUE to extract only features, otherwise get GR value as well
        :return:
        """
        return self.getTopFeatures(topNumber=topNumber, onlyTokens=onlyTokens, method='gain_ratio')

    def calculateCosineSim(self, documentList, documentsToken1, documentsToken2):
        """
        Calculate cosine similarity between
        :param documentList: List of shared documents between token 1 and token 2
        :param documentsToken1: List of documents for token 1
        :param documentsToken2: List of documents for token 2
        :return:
        """
        # For each document in shared list, calculate similarity
        totalNum = sum([documentsToken1[docId] * documentsToken2[docId] for docId in documentList])
        totalTokenFrom = sum([documentsToken1[docId] ** 2 for docId in documentsToken1])
        totalTokenTo = sum([documentsToken2[docId] ** 2 for docId in documentsToken2])

        # Verify denominator part is not zero to perform calculation
        totalDen = totalTokenFrom * totalTokenTo

        if totalDen > 0:
            return float(totalNum) / math.sqrt(totalDen)

        return 0

    def calculateAllSimilarities(self):
        """
        Calculate similarities between tokens. This method stores values with no redundancy, i.e.
        for similarities values sim(a,b) = sim(b,a) is only stored once.

        Also, for similarities equal to zero, value is not stored. At the end of the method,
        resulting file is stored.

        Final storage will return the inverted similarity 1/sim(a,b)
        :return:
        """
        print '[Calculating similarities between tokens]'
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

                            similarityMatrix[tokenFrom][tokenTo] = round(float(1) / cosineSim, 4)

        # Store similarities in class
        self.similarityMatrix = similarityMatrix

        # Store calculation results in file
        with open(self.dictionaryPath + fileconfig.similarityFileName, 'w') as similarityFile:
            similarityFile.write(json.dumps(similarityMatrix, separators=(',', ':')))
            similarityFile.close()

    def loadSimilarities(self):
        """
        Load similarity values calculations
        :return:
        """
        print '[Loading similarity values]'
        similarityMatrixFilePath = self.dictionaryPath + fileconfig.similarityFileName
        if os.path.exists(similarityMatrixFilePath):
            with open(similarityMatrixFilePath, 'r') as similarityFile:
                self.similarityMatrix = json.loads(similarityFile.read())
                similarityFile.close()

    def freeSimilarities(self):
        """
        Free memory for similarity matrix
        :return:
        """
        del (self.similarityMatrix)
        self.similarityMatrix = {}

    def calculateSimilarity(self, token1, token2):
        """
        Calculate similarities between tokens and a list of tokens
        :param token: Token to check similarity
        :param tokenList: List of tokens to compare
        :return: Most similar token
        """
        # Verify if token exists in dictionary
        if token1 in self.postingDocuments:
            # Get list of documents and total of repetitions of the token
            documentFrom = self.postingDocuments[token1]

            # Get document list as set
            documentFromItems = set(documentFrom.keys())

            if token2 in self.postingDocuments:
                # Get list of documents to other tokens
                documentTo = self.postingDocuments[token2]
                similarDocuments = documentFromItems.intersection(documentTo.keys())

                # If there are similar documents, calculate cosine similarity
                if len(similarDocuments) > 0:
                    cosineSim = self.calculateCosineSim(documentList=similarDocuments,
                                                        documentsToken1=documentFrom,
                                                        documentsToken2=documentTo)

                    if cosineSim > 0:
                        return round(float(1) / cosineSim, 4)

        # In case token does not exist, return none
        return 0

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
        return 0

    def saveToDisk(self, calculateSimilarities=True):
        """
        Save index content to disk
        :return:
        """
        print '[Storing data for dictionary: ' + self.dictionaryName + ']'

        # Create folder (if needed)
        self.createDictionaryPath()

        # Dump index stats
        with open(self.dictionaryPath + fileconfig.indexStatsFileName, 'w') as indexStatsFile:
            indexStatsFile.write(json.dumps({
                'terms': self.termCount,
                'documents': self.documentCount,
                'categories': self.categories
            }, separators=(',', ':')))
            indexStatsFile.close()

        # Dump posting documents
        with open(self.dictionaryPath + fileconfig.postingDocsFileName, 'w') as postingDocumentsFile:
            postingDocumentsFile.write(json.dumps(self.postingDocuments, separators=(',', ':')))
            postingDocumentsFile.close()

        # Dump document list
        with open(self.dictionaryPath + fileconfig.documentsFileName, 'w') as documentsFile:
            documentsFile.write(json.dumps(self.documents, separators=(',', ':')))
            documentsFile.close()

        # Dump postings
        with open(self.dictionaryPath + fileconfig.postingsFileName, 'w') as postingsFile:
            postingsFile.write(json.dumps(self.postings, separators=(',', ':')))
            postingsFile.close()

        # Calculate TF-IDF
        print '[Calculating TF-IDF]'
        self.calculateTfIdf()

        # Dump tf values
        with open(self.dictionaryPath + fileconfig.tfFileName, 'w') as tfFile:
            tfFile.write(json.dumps(self.tf, separators=(',', ':')))
            tfFile.close()

        # Dump idf values
        with open(self.dictionaryPath + fileconfig.idfFileName, 'w') as idfFile:
            idfFile.write(json.dumps(self.idf, separators=(',', ':')))
            idfFile.close()

        # Dump tfidf values
        with open(self.dictionaryPath + fileconfig.tfidfFileName, 'w') as tfidfFile:
            tfidfFile.write(json.dumps(self.tfidf, separators=(',', ':')))
            tfidfFile.close()

        # Calculate Information Gain and Gain Ratio for tokens and store in disk
        print '[Calculating Information Gain and Gain Ratio]'
        self.calculateInformationGainAndGainRatio()

        with open(self.dictionaryPath + fileconfig.informationGainFileName, 'w') as igFile:
            igFile.write(json.dumps(self.tokenInfoGain, separators=(',', ':')))
            igFile.close()

        with open(self.dictionaryPath + fileconfig.gainRatioFileName, 'w') as grFile:
            grFile.write(json.dumps(self.tokenGainRatio, separators=(',', ':')))
            grFile.close()

        # Calculate token similarities
        if calculateSimilarities is True:
            self.calculateAllSimilarities()

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
        else:
            return False

        # Get posting documents
        postingDocsFilePath = self.dictionaryPath + fileconfig.postingDocsFileName
        if os.path.exists(postingDocsFilePath):
            with open(postingDocsFilePath, 'r') as postingDocumentsFile:
                self.postingDocuments = json.loads(postingDocumentsFile.read())
                postingDocumentsFile.close()
        else:
            return False

        # Get document list
        documentFilePath = self.dictionaryPath + fileconfig.documentsFileName
        if os.path.exists(documentFilePath):
            with open(documentFilePath, 'r') as documentsFile:
                self.documents = json.loads(documentsFile.read())
                documentsFile.close()
        else:
            return False

        # Get postings
        postingsFilePath = self.dictionaryPath + fileconfig.postingsFileName
        if os.path.exists(postingsFilePath):
            with open(postingsFilePath, 'r') as postingsFile:
                self.postings = json.loads(postingsFile.read())
                postingsFile.close()
        else:
            return False

        # Get tfidf values
        tfidfFilePath = self.dictionaryPath + fileconfig.tfidfFileName
        if os.path.exists(tfidfFilePath):
            with open(tfidfFilePath, 'r') as tfidfFile:
                self.tfidf = json.loads(tfidfFile.read())
                tfidfFile.close()

        # Get information gain values
        igFilePath = self.dictionaryPath + fileconfig.informationGainFileName
        if os.path.exists(igFilePath):
            with open(igFilePath, 'r') as igFile:
                self.tokenInfoGain = json.loads(igFile.read())

                # Sort token by highest information gain
                self.orderedTokenInfoGain = sorted(
                    self.tokenInfoGain,
                    key=self.tokenInfoGain.__getitem__,
                    reverse=True
                )

                # Close file
                igFile.close()

        # Get gain ratio values
        gainRatioFilePath = self.dictionaryPath + fileconfig.gainRatioFileName
        if os.path.exists(gainRatioFilePath):
            with open(gainRatioFilePath, 'r') as gainRatioFile:
                self.tokenGainRatio = json.loads(gainRatioFile.read())
                gainRatioFile.close()

            # Sort token by highest information gain
            self.orderedTokenGainRatio = sorted(
                self.tokenGainRatio,
                key=self.tokenGainRatio.__getitem__,
                reverse=True
            )

        return True

    def createArffFile(self, arffFileName, tokenList=[]):
        """
        Create ARFF file from dictionary for Weka classification
        :param arffFileName: File name to store data
        :param tokenList: Token list to use
        :return:
        """

        # Verify if token list has items to use them, otherwise use all postings
        if len(tokenList) == 0:
            tokenList = set(self.postings)

        with open(dirconfig.arffPath + arffFileName + '.arff', 'w') as arffFile:
            # Step 1: header information for ARFF file
            arffFile.write("@relation 'docs-" + self.dictionaryName + "'\n")

            # Add attribute list (tokens to use in classification)
            for token in tokenList:
                token = token.encode('utf-8')
                token = re.escape(token)
                arffFile.write("@attribute '" + token + "' {n,y}\n")

            # Add class attribute and list of different classes
            categoryList = []

            for category in sorted(self.categories.keys()):
                categoryList.append(re.escape(category))

            arffFile.write("@attribute 'docClass' {" + ','.join(categoryList) + "}\n")

            # Add data attribute
            arffFile.write("\n@data\n")

            # Step 2: include data values for each document
            for docId in self.documents:

                # List of document items [tokens + class]
                documentItems = []

                # get document tokens
                documentTokens = self.documents[docId]['t']

                for token in tokenList:
                    if token in documentTokens:
                        # Whether the token is in the document, add  'y' value.
                        documentItems.append('y')
                    else:
                        # Token is not on document, add 'n' value
                        documentItems.append('n')

                # Add class to document items
                documentItems.append(re.escape(self.documents[docId]['c']))

                # Write document items in file
                arffFile.write(','.join(documentItems) + '\n')

            arffFile.close()
