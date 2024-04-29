"""
Code Written by project Telekinesis Team: Gabriel Koeller, Jaden Hicks, and Sydney Rash

Adapted from the example file neurofeedback.py found at https://github.com/alexandrebarachant/muse-lsl/blob/f899ba3b7e8c54bb73630f478e114d724bf715f0/examples/neurofeedback.py
Alexandre Barachant, Dano Morrison, Hubert Banville, Jason Kowaleski, Uri Shaked, Sylvain Chevallier, & Juan Jesús Torre Tresols. (2019, May 25). muse-lsl (Version v2.0.2). Zenodo. http://doi.org/10.5281/zenodo.3228861

Also adapted from ChilloutCharles BrainFlowsIntoVRChat found at https://github.com/ChilloutCharles/BrainFlowsIntoVRChat
ChilloutCharles can be found at: https://linktr.ee/ChilloutCharles
"""

#Libraries
import pylsl
import numpy
import utils #utils.py must be in the same folder
import pickle


"""================================================================================================================================="""


"""VARIABLE SETUP"""

#Parameters
BATCH_LENGTH = 5 #Length of data to record in seconds

MINI_LENGTH = 1 #Length of mini-batch to use in calculations

OVERLAP = .9 #Amount overlap between mini-batches in seconds

SHIFT_LENGTH = MINI_LENGTH - OVERLAP #Amount to shift in data to grab next mini-batch

#Used to index sensors if using all 5 is not wanted
#Left-most is 0, the right-most is 3, AUX is 4
SENSOR_CHANNEL = [0, 1, 2, 3]
SENSORS = len(SENSOR_CHANNEL)

#Used to keep track of time while recording data
timeStep = 0 #Used to keep track of how many data points have been recorded.
actionNum = 0 #Used to label data for supervised learning.
maxTimeSteps = int(60 / SHIFT_LENGTH) #Calulate how many minibatches are in 1 minute (60 seconds).
#A minibatch is created/updated every time a shift length's amount of data is recieved


"""================================================================================================================================="""


"""EEG SETUP"""

#Begin connecting to device (Stream needs to already be active)

devices = pylsl.resolve_byprop('type', 'EEG', timeout=10)
if len(devices) == 0:
    raise RuntimeError("No Muses found.\nPlease ensure bluetooth is enabled and the device is turned on.")

#Begin device stream
stream = pylsl.StreamInlet(devices[0]) #0 means the first device found if multiple are available

#Get stream sampling frequency
freq = int(stream.info().nominal_srate())
#This is used to determine how many data points represent 1 second so we can grab data from a specific time period

#Create a buffer to hold the data from the stream
#BATCH_LENGTH is the length in seconds, and freq is the number of datapoints per second, creating datapoints per batch when multiplied
#The Muse 2 uses 5 sensors
streamBatch = numpy.zeros((int(freq * BATCH_LENGTH), SENSORS))
filterState = None #Used for notch filter later

miniBatchesPerBatch = int(numpy.floor((BATCH_LENGTH - MINI_LENGTH) / SHIFT_LENGTH + 1))
bandBuffer = numpy.zeros((miniBatchesPerBatch, 5)) #Bands are frequency groups representing the different brainwaves
#The bands are Delta, Theta, Alpha, Beta, Gamma.
#Delta has the lowest hertz and is associated with the unconscious brain, whereas Gamma has the highest hertz and is associated with cognitive processing 


"""================================================================================================================================="""


"""USER SETUP"""

print("How many actions (including a null action) would you like to train?")
ACTIONS = input("Num Actions: ")
while not ACTIONS.isdigit():
    print("Please enter a positive integer.")
    ACTIONS = input("Num Actions: ")
ACTIONS = int(ACTIONS)

#Dictionary will be used to sort between actions
#Each action will have an entry in the dictionary with a list holding recorded data associated with the action
actionDict = {
    actionId:[] for actionId in range(ACTIONS)
}


"""================================================================================================================================="""


"""RECORD DATA"""

while actionNum < ACTIONS:
        
    if timeStep == 0:
        print("Please begin envisioning action " + str(actionNum+1) + ". Data will be collected for approx. 1 minute.")
        input("Press the enter key when ready.")
        print()
        print()
        print("Collecting Data...")
        print()
        print()
        
    #Obtain data from the stream
    data, timeStamp = stream.pull_chunk(timeout=3, max_samples=int(SHIFT_LENGTH * freq))
    
    #Convert data to numpy array for functional purposes
    #dataArray = numpy.array(data)
    #To restrict the data to only chosen sensors, use this line instead
    dataArray = numpy.array(data)[:, SENSOR_CHANNEL]
    
    #Update data buffer
    streamBatch, filterState = utils.update_buffer(streamBatch, dataArray, notch=True, filter_state=filterState)
    
    #Create mini-batch
    miniBatch = utils.get_last_data(streamBatch, int(MINI_LENGTH * freq))
    #print(miniBatch)
    
    #Compute band powers and store them in a buffer
    bandPowers = utils.compute_band_powers(miniBatch, freq)
    bandBuffer, _ = utils.update_buffer(bandBuffer, numpy.asarray(bandPowers))
    
    #Average out each band to reduce noise
    smoothBandPowers = numpy.mean(bandBuffer, axis=0) #Remember axis 0 is the rows, so you take the average across all epochs stored in the buffer.
    
    #Save data to dictionary
    actionDict[actionNum].append(smoothBandPowers)
    
    #Update timeStep
    timeStep = timeStep + 1
    #print(timeStep)
    if timeStep > maxTimeSteps:
        timeStep = 0
        actionNum = actionNum + 1


"""================================================================================================================================="""


"""SAVE EVERYTHING"""

saveName = input("Please enter a profile name:")
saveName = saveName + ".pkl"
print("Saving data and ending program")

#Save the pickle
with open(saveName, "wb") as pklFile:
    pickle.dump(actionDict, pklFile)

print("Data saved")