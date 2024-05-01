# ProjectTelekinesis
Code used for Senior Capstone project "Project Telekinesis"
This code can be used to train an AI model on intended actions using a Muse 2 EEG.

Code Written by project Telekinesis Team: Gabriel Koeller, Jaden Hicks, and Sydney Rash

Adapted from the example file neurofeedback.py found at https://github.com/alexandrebarachant/muse-lsl/blob/f899ba3b7e8c54bb73630f478e114d724bf715f0/examples/neurofeedback.py
Alexandre Barachant, Dano Morrison, Hubert Banville, Jason Kowaleski, Uri Shaked, Sylvain Chevallier, & Juan Jesús Torre Tresols. (2019, May 25). muse-lsl (Version v2.0.2). Zenodo. http://doi.org/10.5281/zenodo.3228861

Also adapted from ChilloutCharles BrainFlowsIntoVRChat found at https://github.com/ChilloutCharles/BrainFlowsIntoVRChat
ChilloutCharles can be found at: https://linktr.ee/ChilloutCharles

If a user has never used Telekinesis before, the first program to run is TeleDataPKL.py. This program will collect data for each intended action from the EEG.
After running TeleDataPKL.py, TeleRNNPKL.py should be run to train an AI on the recorded data.
If a user has trained an AI on their data, TeleRoverConnection should be run to connect the EEG to the rover through the AI.
