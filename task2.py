import numpy as np
import tensorflow as tf
from collections import defaultdict

# the number of iterations to train for
numTrainingIters = 10000
# the number of iterations to test for
numTestingIters = 30
# the number of hidden neurons that hold the state of the RNN
hiddenUnits = 500
# the number of classes that we are learning over
numClasses = 3
# the number of data points in a batch
batchSize = 100

def addToDataAndTestNoReplacement(maxSeqLen, data, fileName, classNum, linesToUse, testData, numTestLines):
    #
    # open the file and read it in
    with open(fileName) as f:
        content = f.readlines()
    # sample linesToUse numbers; these will tell us what lines
    # from the text file we will use
    myInts = np.random.choice(np.arange(0,linesToUse), linesToUse, replace = False)
    # i is the key of the next line of text to add to the dictionary
    i = len(data)
    testDataKey = len(testData)
    # loop thru and add the lines of text to the dictionary
    numTestSoFar = 0
    ind = 0
    sampleInds = myInts.flat
    while numTestSoFar < numTestLines:
        whichLine = sampleInds[ind]
        ind += 1
        #
        # get the line and ignore it if it has nothing in it
        line = content[whichLine]
        if line.isspace () or len(line) == 0:
            continue
        #
        # take note if this is the longest line we've seen
        if len (line) > maxSeqLen:
            maxSeqLen = len (line)
        #
        # create the matrix that will hold this line
        temp = np.zeros((len(line), 256))
        #
        # j is the character we are on
        j = 0
        #
        # loop thru the characters
        for ch in line:
            #
            # non-ascii? ignore
            if ord(ch) >= 256:
                continue
            #
            # one hot!
            temp[j][ord(ch)] = 1
            #
            # move onto the next character
            j = j + 1
            #
        # remember the line of text
        testData[testDataKey] = (classNum, temp)
        # move onto the next line
        testDataKey += 1
        numTestSoFar += 1
    for whichLine in sampleInds[ind:]:
        #
        # get the line and ignore it if it has nothing in it
        line = content[whichLine]
        if line.isspace () or len(line) == 0:
            continue
        #
        # take note if this is the longest line we've seen
        if len (line) > maxSeqLen:
            maxSeqLen = len (line)
        #
        # create the matrix that will hold this line
        temp = np.zeros((len(line), 256))
        #
        # j is the character we are on
        j = 0
        #
        # loop thru the characters
        for ch in line:
            #
            # non-ascii? ignore
            if ord(ch) >= 256:
                continue
            #
            # one hot!
            temp[j][ord(ch)] = 1
            #
            # move onto the next character
            j = j + 1
            #
        # remember the line of text
        data[i] = (classNum, temp)
        #
        # move onto the next line
        i = i + 1
        #
        # and return the dictionaries with the new data
    return (maxSeqLen, data, testData)


def pad (maxSeqLen, data):
   #
   # loop thru every line of text
   for i in data:
        #
        # access the matrix and the label
        temp = data[i][1]
        label = data[i][0]
        # 
        # get the number of chatacters in this line
        len = temp.shape[0]
        #
        # and then pad so the line is the correct length
        padding = np.zeros ((maxSeqLen - len,256)) 
        data[i] = (label, np.transpose (np.concatenate ((padding, temp), axis = 0)))
   #
   # return the new data set
   return data

def getDataLines(isTest, batSize):
    """
    Gets a batch of data lines of size batSize. 
    Intended to get the lines without replacement.
    :param isTest: whether this is from the test data or not. 
    :param batSize: size of the batch
    :return: 
    """
    global nextUnusedTestLine, nextUnusedTrainLine
    if isTest:
        if nextUnusedTestLine % len(testData) > (nextUnusedTestLine + batSize) % len(testData):
            nextUnusedTestLine = 0
        myInts = np.arange(nextUnusedTestLine % len(testData), (nextUnusedTestLine + batSize) % len(testData))
        nextUnusedTestLine += batSize
    else:
        if nextUnusedTrainLine % len(data) > (nextUnusedTrainLine + batSize) % len(data):
            nextUnusedTrainLine = 0
        myInts = np.arange(nextUnusedTrainLine % len(data), (nextUnusedTrainLine + batSize) % len(data))
        nextUnusedTrainLine += batSize
    return myInts


def generateDataRNN (maxSeqLen, data, isTest):
    #
    # randomly sample batchSize lines of text
    # myInts = np.random.random_integers (0, len(data) - 1, batchSize)
    myInts =getDataLines(isTest, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack (data[i][1] for i in myInts.flat)
    #
    # and stack all of the labels into a vector of labels
    y = np.stack (np.array((data[i][0])) for i in myInts.flat)
    #
    # return the pair
    return (x, y)


# create the data dictionary
maxSeqLen = 0
data = {}
testData = {}
numTestLines = 1000

trainLines = data.keys()
testLines = testData.keys()
nextUnusedTestLine = 0
nextUnusedTrainLine = 0

# load up the three data sets
(maxSeqLen, data, testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "Holmes.txt", 0, 11000, testData, numTestLines)
(maxSeqLen, data, testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "war.txt", 1, 11000, testData, numTestLines)
(maxSeqLen, data, testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "william.txt", 2, 11000, testData, numTestLines)

# pad each entry in the dictionary with empty characters as needed so
# that the sequences are all of the same length
data = pad (maxSeqLen, data)
testData = pad (maxSeqLen, testData)
# note: we just pad the data and test data to same amount to be safe

# now we build the TensorFlow computation... there are two inputs, 
# a batch of text lines and a batch of labels
inputX = tf.placeholder(tf.float32, [batchSize, 256, maxSeqLen])
inputY = tf.placeholder(tf.int32, [batchSize])

# this is the inital state of the RNN, before processing any data
initialState = tf.placeholder(tf.float32, [batchSize, hiddenUnits])

# the weight matrix that maps the inputs and hidden state to a set of values
W = tf.Variable(np.random.normal(0, 0.05, (256 + hiddenUnits + hiddenUnits, hiddenUnits)), dtype=tf.float32)

# biases for the hidden values
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClasses)),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

# unpack the input sequences so that we have a series of matrices,
# each of which has a one-hot encoding of the current character from
# every input sequence
sequenceOfLetters = tf.unstack(inputX, axis=2) #TODO: comment this out for Task 3

# and train!!
# now we implement the forward pass
initialTimeWarpState = tf.Variable(np.zeros((batchSize, hiddenUnits)), dtype=tf.float32)

def returnInitialTimeWarp():
    return initialTimeWarpState

timeWarpStates = defaultdict(returnInitialTimeWarp)

currentState = initialState
tick = 0
for timeTick in sequenceOfLetters:
    #
    # concatenate the state with the input, then compute the next state
    oldState = timeWarpStates[tick - 10]
    inputPlusState = tf.concat([timeTick, currentState, oldState], 1)
    timeWarpStates[tick] = tf.tanh(tf.matmul(inputPlusState, W) + b)
    currentState = timeWarpStates[tick]
    tick += 1

# compute the set of outputs
outputs = tf.matmul(currentState, W2) + b2
predictions = tf.nn.softmax(outputs)
# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)
# use gradient descent to train
trainingAlg = tf.train.AdagradOptimizer(0.02).minimize(totalLoss)

testingLossSum = 0
correctTestGuess = 0
########## THE ACTUAL RUN: ############
with tf.Session() as sess:
    # initialize everything
    sess.run(tf.global_variables_initializer())
    # and run the training iters
    for epoch in range(numTrainingIters):
        #
        # get some data
        x, y = generateDataRNN (maxSeqLen, data, False)
        #
        # do the training epoch
        _currentState = np.zeros((batchSize, hiddenUnits))
        _totalLoss, _trainingAlg, _currentState, _predictions, _outputs = sess.run(
            [totalLoss, trainingAlg, currentState, predictions, outputs],
            feed_dict={
                inputX: x,
                inputY: y,
                initialState: _currentState
            })
        # just FYI, compute the number of correct predictions
        numCorrect = 0
        for i in range (len(y)):
           maxPos = -1
           maxVal = 0.0
           for j in range (numClasses):
               if maxVal < _predictions[i][j]:
                   maxVal = _predictions[i][j]
                   maxPos = j
           if maxPos == y[i]:
               numCorrect = numCorrect + 1
        # print out to the screen
        if epoch > numTrainingIters - 30:
            print("Step", epoch, "Loss", _totalLoss, "Correct", numCorrect, "out of", batchSize)
    ## TESTING PHASE ##
    #
    # initialize everything
    # sess.run(tf.global_variables_initializer())
    #
    # and run the training iters
    for epoch in range(numTestingIters):
        # TODO: we need some way to get a batch of test data
        x, y = generateDataRNN (maxSeqLen, testData, True)
        _currentState = np.zeros((batchSize, hiddenUnits))
        currTestLoss, testPredictions = sess.run(
            [totalLoss, predictions],
            feed_dict={
                inputX: x,
                inputY: y,
                initialState: _currentState
            })
        correctTestBatchGuess = 0
        testingLossSum += currTestLoss
        for i in range (len(y)):
           maxPos = -1
           maxVal = 0.0
           for j in range (numClasses):
               if maxVal < testPredictions[i][j]:
                   maxVal = testPredictions[i][j]
                   maxPos = j
           if maxPos == y[i]:
               correctTestBatchGuess += 1
               correctTestGuess += 1
        print("Testing Step", epoch, "Loss", currTestLoss, "Correct", correctTestBatchGuess, "out of", batchSize)
        testingLossSum += currTestLoss


print("Loss for 3000 randomly chosen documents is %f, number correct labels is %d out of 3000"
      % (testingLossSum/numTestingIters, correctTestGuess))


### RESULTS ###
# Step 9971 Loss 0.14255996 Correct 96 out of 100
# Step 9972 Loss 0.16208628 Correct 92 out of 100
# Step 9973 Loss 0.26071727 Correct 91 out of 100
# Step 9974 Loss 0.13774651 Correct 94 out of 100
# Step 9975 Loss 0.29318553 Correct 87 out of 100
# Step 9976 Loss 0.21059859 Correct 93 out of 100
# Step 9977 Loss 0.22710264 Correct 90 out of 100
# Step 9978 Loss 0.2586771 Correct 89 out of 100
# Step 9979 Loss 0.13167946 Correct 96 out of 100
# Step 9980 Loss 0.2578965 Correct 92 out of 100
# Step 9981 Loss 0.26793286 Correct 87 out of 100
# Step 9982 Loss 0.23124152 Correct 91 out of 100
# Step 9983 Loss 0.24692452 Correct 91 out of 100
# Step 9984 Loss 0.10854376 Correct 96 out of 100
# Step 9985 Loss 0.21011515 Correct 93 out of 100
# Step 9986 Loss 0.15902333 Correct 91 out of 100
# Step 9987 Loss 0.30943668 Correct 87 out of 100
# Step 9988 Loss 0.23290431 Correct 90 out of 100
# Step 9989 Loss 0.11076446 Correct 95 out of 100
# Step 9990 Loss 0.1917619 Correct 91 out of 100
# Step 9991 Loss 0.26651552 Correct 88 out of 100
# Step 9992 Loss 0.19909286 Correct 92 out of 100
# Step 9993 Loss 0.18254288 Correct 95 out of 100
# Step 9994 Loss 0.2799754 Correct 90 out of 100
# Step 9995 Loss 0.19136761 Correct 92 out of 100
# Step 9996 Loss 0.1504795 Correct 94 out of 100
# Step 9997 Loss 0.17935029 Correct 93 out of 100
# Step 9998 Loss 0.3300157 Correct 86 out of 100
# Step 9999 Loss 0.17571288 Correct 91 out of 100
# TESTING:
# Loss for 3000 randomly chosen documents is 1.229877, number correct labels is 942 out of 3000


