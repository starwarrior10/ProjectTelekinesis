#======================================================================
#  Project Telekinesis - Senior Capstone Project
#
#  Name: Rover.py
#  Purpose: create interface to connect to Arduino Uno R3 and control
#       a rover
#  Preconditions: the Arduino has the program `prototype.ino` uploaded
#       from the Arduino Python library GitHub. The motors are
#       connected to the Arduino through the Elegoo
#       SmartCar-Sheild-V1.1.
#
#  To use this program,
#    1. Install the arduino-python3 library.
#          `pip install arduino-python 3`
#    2. Connect the computer to the Arduino Uno via a USB cord.
#    3. Create an instance of the rover class.
#          `rover = Rover()` # automatically tries to connects to rover
#    4. Call `rover.test()` to test movement and `rover.move(dir)`
#       to directly control the rover. 
#======================================================================

"""
Code Written by project Telekinesis Team: Gabriel Koeller, Jaden Hicks, and Sydney Rash

This code is used to control a rover through a previously trained RNN

Built by self and partially adapted from ChilloutCharles BrainFlowsIntoVRChat found at https://github.com/ChilloutCharles/BrainFlowsIntoVRChat
ChilloutCharles can be found at: https://linktr.ee/ChilloutCharles
"""
        
#Libraries
import pylsl
import numpy
import utils #utils.py must be in the same folder
import keras
import Arduino # library: arduino-python3


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

model = keras.models.load_model("TestGRUModel.keras")


"""================================================================================================================================="""


"""ROVER SETUP"""


class Rover:
  def __init__(self, speed = 100):
    self.speed = speed
    # Set pin numbers
    self.PIN_Motor_STBY = 3 # enables motor control
    self.PIN_Motor_PWMA = 5 # analog, controls Motor A speed
    self.PIN_Motor_PWMB = 6 # analog, controls Motor B speed
    self.PIN_Motor_BIN_1 = 7 # digital, controls motor B direction
    self.PIN_Motor_AIN_1 = 8 # digital, controls motor A direction

    # Connect to board through USB
    self.connect() 

    # Set pin modes
    self.board.pinMode(self.PIN_Motor_PWMA, "OUTPUT");
    self.board.pinMode(self.PIN_Motor_PWMB, "OUTPUT");
    self.board.pinMode(self.PIN_Motor_AIN_1, "OUTPUT");
    self.board.pinMode(self.PIN_Motor_BIN_1, "OUTPUT"); 
  
    # Set speed
    self.board.analogWrite(self.PIN_Motor_PWMA, self.speed);
    self.board.analogWrite(self.PIN_Motor_PWMB, self.speed);

  def connect(self, port="9600"):
    # Connect to Arduino Uno board
    self.board = Arduino.Arduino(port) # plugged in via USB, serial com at rate 9600

  def move(self, direction):
    # Move the rover
    #    direction: 2 BRAKE, 0 FORWARD, 1 BACKWARD
    if direction == 2:    # BRAKE
      # Disable motor control
      self.board.digitalWrite(self.PIN_Motor_STBY, "LOW");
    else:                 # MOVE
      # Enable motor control
      self.board.digitalWrite(self.PIN_Motor_STBY, "HIGH");
      # Set direction
      if direction == 0:  # FORWARD
        self.board.digitalWrite(self.PIN_Motor_AIN_1, "HIGH");
        self.board.digitalWrite(self.PIN_Motor_BIN_1, "HIGH");
      else:               # BACKWARD
        self.board.digitalWrite(self.PIN_Motor_AIN_1, "LOW");
        self.board.digitalWrite(self.PIN_Motor_BIN_1, "LOW");
        
rover = Rover()
        

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


"""OBTAIN DATA AND PREDICT"""


print("Ctrl + c can be pressed to kill the program at any time.")

#Create buffer for AI input
predictionBufferSize = int(MINI_LENGTH/SHIFT_LENGTH)
predictionOldBuffer = numpy.ones((predictionBufferSize, 5)) #Used to store the old data for transfer
predictionNewBuffer = numpy.ones((predictionBufferSize, 5)) #Holds newest input and is used by the AI
predictionShape = predictionNewBuffer.shape #Calculating once for later in the loop

while(True):
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
    
    #Update prediction buffers
    predictionOldBuffer = predictionNewBuffer
    predictionNewBuffer[0:predictionBufferSize-1] = predictionOldBuffer[1:]
    predictionNewBuffer[-1] = smoothBandPowers
    
    #Here goes nothing!
    chosenAction = numpy.argmax(model(predictionNewBuffer.reshape(1, predictionShape[0], predictionShape[1]))) #Predict requires (NumSample, NumFeatures) so we add a dimension here
    
    print(chosenAction) #Debugging
    
    if chosenAction == 0: #Debugging
        print("Forward")
    elif chosenAction == 1:
        print("Backward")
    else:
        print("Stop")
    
    rover.move(chosenAction) #Move the connected rover