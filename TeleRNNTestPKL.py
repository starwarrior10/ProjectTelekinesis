"""
Code Written by project Telekinesis Team: Gabriel Koeller, Jaden Hicks, and Sydney Rash

This code is intended to test a RNN (GRU) network on test data

Built by self and partially adapted from ChilloutCharles BrainFlowsIntoVRChat found at https://github.com/ChilloutCharles/BrainFlowsIntoVRChat
ChilloutCharles can be found at: https://linktr.ee/ChilloutCharles
"""

import numpy
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

with open("TestGRUModel.pkl", "rb") as pklFile:
    dataDict = pickle.load(pklFile)
    
NUM_ACTIONS = len(dataDict)
_, NUM_FEATURES = numpy.array(dataDict[0]).shape

#Parameters
BATCH_LENGTH = 5 #Length of data to record in seconds

MINI_LENGTH = 1 #Length of mini-batch to use in calculations

OVERLAP = .9 #Amount overlap between mini-batches in seconds

SHIFT_LENGTH = MINI_LENGTH - OVERLAP #Amount to shift in data to grab next mini-batch

#Define chunk size as 1 second with an overlap of .9 seconds
CHUNK_LENGTH = int(MINI_LENGTH / SHIFT_LENGTH)
#Minibatch length is time in seconds so by dividing that by the shift length, we get the exact amount of datapoints we need to get 1-overlap seconds of new data

def chunkData(data, chunkSize, overlap):
    numDataPoints, _ = data.shape #Since data will be a 2d array, we only need the number data points, not the attributes of each datapoint
    stepLength = chunkSize - overlap
    chunks = []
    
    for start in range(0, ((numDataPoints - chunkSize) + 1), stepLength): #From 0 to the start of the final chunk
        end = start + chunkSize
        chunk = data[start:end, :]
        chunks.append(chunk)
    
    return numpy.array(chunks) #Behold, a 3d array, or an array of 2d arrays

def createChunks(data):
    actionData = numpy.array([entry for entry in data])
    chunksAction = chunkData(actionData, CHUNK_LENGTH, int(OVERLAP * CHUNK_LENGTH)) #Overlap is multiplied by chunk length to get the number of entries to overlap
    chunks = numpy.concatenate([chunksAction]) #Needs to be in brackets othewise it unchunks the data and puts it all in as one chunk
    return chunks

inputChunks = {a:createChunks(d) for a, d in dataDict.items()}

labels = numpy.concatenate([[a] * len(d) for a, d in inputChunks.items()]) #Create an array with the action for each chunk to serve as a label
inputData = numpy.concatenate(list(inputChunks.values()))
inputTrain, inputTest, labelTrain, labelTest = train_test_split(inputData, labels, test_size=0.2, stratify=labels)


GRUModel = keras.Sequential(
    [
     keras.layers.InputLayer(input_shape=(CHUNK_LENGTH, NUM_FEATURES)),
     keras.layers.GRU(32, return_sequences=True), #Return sequence outputs the whole sequence instead of just the last entry in the sequence
     keras.layers.GRU(64, return_sequences=True),
     keras.layers.GRU(128),
     keras.layers.Dense(NUM_ACTIONS, activation="Softmax")
     ]
)

GRUModel.summary()

#Compile the model
GRUModel.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

#Train and test the model
modelHistory = GRUModel.fit(
    inputTrain,
    labelTrain,
    batch_size=10,
    epochs=100,
    validation_data=(inputTest, labelTest)
)

plt.plot(modelHistory.history["loss"])
plt.plot(modelHistory.history["val_loss"])
plt.plot(modelHistory.history["sparse_categorical_accuracy"])
plt.plot(modelHistory.history["val_sparse_categorical_accuracy"])
plt.title("Model Acc. and Loss")
plt.xlabel("Epoch")
plt.ylabel("Acc., Loss")
plt.legend(["Train Loss", "Val Loss", "Train Acc.", "Val Acc."], loc="lower left")

print("Accuracy: " + str(modelHistory.history["val_sparse_categorical_accuracy"][-1]*100) + "%")

GRUModel.save("TestGRUModel.keras")