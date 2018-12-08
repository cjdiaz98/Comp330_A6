import numpy as np
import tensorflow as tf
from collections import defaultdict

# the number of iterations to train for
numTrainingIters = 10000

# the number of hidden neurons that hold the state of the RNN
# hiddenUnits = 500 # TODO: make this 500 for task 2
hiddenUnits = 1000


# the number of classes that we are learning over
numClasses = 3

# the number of data points in a batch
batchSize = 100

# one hot  encoding -- character from 0 to 255, fancy name for dictionary encoding. gets the line

# this function takes a dictionary (called data) which contains 
# of (dataPointID, (classNumber, matrix)) entries.  Each matrix
# is a sequence of vectors; each vector has a one-hot-encoding of
# an ascii character, and the sequence of vectors corresponds to
# one line of text.  classNumber indicates which file the line of
# text came from.  
# 
# The argument maxSeqLen is the maximum length of a line of text
# seen so far.  fileName is the name of a file whose contents
# we want to add to data.  classNum is an indicator of the class
# we are going to associate with text from that file.  linesToUse
# tells us how many lines to sample from the file.
#
# The return val is the new maxSeqLen, as well as the new data
# dictionary with the additional lines of text added
def addToData (maxSeqLen, data, fileName, classNum, linesToUse):
    #
    # open the file and read it in
    with open(fileName) as f:
        content = f.readlines()
    #
    # sample linesToUse numbers; these will tell us what lines
    # from the text file we will use
    myInts = np.random.random_integers (0, len(content) - 1, linesToUse)
    #
    # i is the key of the next line of text to add to the dictionary
    i = len(data)
    #
    # loop thru and add the lines of text to the dictionary

    for whichLine in myInts.flat:

        #
        # get the line and ignore it if it has nothing in it
        line = content[whichLine]
        if line.isspace () or len(line) == 0:
            continue;
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
    # and return the dictionary with the new data
    return (maxSeqLen, data)


def addToDataNoReplacement(maxSeqLen, data, fileName, classNum, linesToUse):
    #
    # open the file and read it in
    with open(fileName) as f:
        content = f.readlines()
    #
    # sample linesToUse numbers; these will tell us what lines
    # from the text file we will use
    myInts = np.random.choice(np.arange(0,linesToUse), linesToUse, replace = False)

    #
    # i is the key of the next line of text to add to the dictionary
    i = len(data)
    #
    # loop thru and add the lines of text to the dictionary

    for whichLine in myInts.flat:

        #
        # get the line and ignore it if it has nothing in it
        line = content[whichLine]
        if line.isspace () or len(line) == 0:
            continue;
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
    # and return the dictionary with the new data
    return (maxSeqLen, data)


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
            continue;
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
    for whichLine in  sampleInds[ind:]:
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


# this function takes as input a data set encoded as a dictionary
# (same encoding as the last function) and pre-pends every line of
# text with empty characters so that each line of text is exactly
# maxSeqLen characters in size
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

# this generates a new batch of training data of size batchSize from the
# list of lines of text data. This version of generateData is useful for
# an RNN because the data set x is a NumPy array with dimensions
# [batchSize, 256, maxSeqLen]; it can be unstacked into a series of
# matrices containing one-hot character encodings for each data point
# using tf.unstack(inputX, axis=2)
def generateDataRNN (maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.random_integers (0, len(data) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack (data[i][1] for i in myInts.flat)
    #
    # and stack all of the labels into a vector of labels
    y = np.stack (np.array((data[i][0])) for i in myInts.flat)
    #
    # return the pair
    return (x, y)

# this also generates a new batch of training data, but it represents
# the data as a NumPy array with dimensions [batchSize, 256 * maxSeqLen]
# where for each data point, all characters have been appended.  Useful
# for feed-forward network training
def generateDataFeedForward (maxSeqLen, data):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.random_integers (0, len(data) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x = np.stack (data[i][1].flatten () for i in myInts.flat)
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

# load up the three data sets
(maxSeqLen, data,testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "Holmes.txt", 0, 11000, testData, numTestLines)
(maxSeqLen, data,testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "war.txt", 1, 11000, testData, numTestLines)
(maxSeqLen, data,testData) = addToDataAndTestNoReplacement (maxSeqLen, data, "william.txt", 2, 11000, testData, numTestLines)

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
W = tf.Variable(np.random.normal(0, 0.05, (hiddenUnits + 256, hiddenUnits)), dtype=tf.float32)

# biases for the hidden values
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClasses)),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

# unpack the input sequences so that we have a series of matrices,
# each of which has a one-hot encoding of the current character from
# every input sequence
sequenceOfLetters = tf.unstack(inputX, axis=2)


# and train!!
############# Task0 Below:   ###############
currentState = initialState
for timeTick in sequenceOfLetters:
    #
    # concatenate the state with the input, then compute the next state
    inputPlusState = tf.concat([timeTick, currentState], 1)
    nextState = tf.tanh(tf.matmul(inputPlusState, W) + b)
    currentState = nextState


############# Task2 Below:   ###############
# now we implement the forward pass
initialTimeWarpState = tf.Variable((tf.float32, [batchSize, hiddenUnits]))
def returnInitialTimeWarp():
    return initialTimeWarpState
timeWarpStates = defaultdict(returnInitialTimeWarp)


currentState = initialTimeWarpState
for timeTick in sequenceOfLetters:
    #
    # concatenate the state with the input, then compute the next state
    inputPlusState = tf.concat([timeTick, currentState], 1)
    timeWarpStates[timeTick + 10] = tf.tanh(tf.matmul(inputPlusState, W) + b)
    currentState = timeWarpStates[timeTick + 1]

####### TASK 2 SPECIFIC CODE OVER  ########

# compute the set of outputs
outputs = tf.matmul(currentState, W2) + b2

predictions = tf.nn.softmax(outputs)

# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)

# use gradient descent to train
#trainingAlg = tf.train.GradientDescentOptimizer(0.02).minimize(totalLoss)
trainingAlg = tf.train.AdagradOptimizer(0.02).minimize(totalLoss)


########## THE ACTUAL RUN: ############
with tf.Session() as sess:
    # initialize everything
    sess.run(tf.global_variables_initializer())
    # and run the training iters
    for epoch in range(numTrainingIters):
        #
        # get some data
        x, y = generateDataRNN (maxSeqLen, data)
        #
        # do the training epoch
        _currentState = np.zeros((batchSize, hiddenUnits))
        # TODO: DO WE MERELY CHANGE THE ABOVE TENSOR FOR TIMEWARP??
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


### FOR TASK 2: ###
# _totalLoss, _trainingAlg, _currentState, _predictions, _outputs, _initialTimeWarpState = sess.run(
#         [totalLoss, trainingAlg, currentState, predictions, outputs, initialTimeWarpState],
#         feed_dict={
#             inputX:x,
#             inputY:y,
#             initialState:_currentState
#         })


## TESTING PHASE ##
numTestingIters = 30
testingLossSum = 0
correctTestGuess = 0

with tf.Session() as sess:
    #
    # initialize everything
    sess.run(tf.global_variables_initializer())
    #
    # and run the training iters
    for epoch in range(numTestingIters):

        # TODO: we need some way to get a batch of test data
        x, y = generateDataRNN (maxSeqLen, testData)
        currTestLoss, testPredictions = sess.run(
            [totalLoss, predictions],
            feed_dict={
                inputX: x,
                inputY: y,
                initialState: _currentState
            })

        for i in range (len(y)):
           maxPos = -1
           maxVal = 0.0
           for j in range (numClasses):
               if maxVal < testPredictions[i][j]:
                   maxVal = testPredictions[i][j]
                   maxPos = j
           if maxPos == y[i]:
               correctTestGuess += 1

        testingLossSum += currTestLoss


print("Loss for 3000 randomly chosen documents is %f, number correct labels is %d out of 3000"
      % (testingLossSum, correctTestGuess))


