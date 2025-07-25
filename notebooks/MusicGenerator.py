import numpy as np

if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

import logging
import logging.config
import tensorflow.keras as keras
import yaml
import json

class MusicGenerator:
    def __init__(self, modelPath):
        log.info("Loading the model from path: %s", modelPath)
        self.modelPath = modelPath
        self.model = keras.models.load_model(self.modelPath)
        log.info("Model loaded successfully.")
        
        # Load the mapping file
        log.info("Loading the mapping file from path: %s", modelPath)
        mappingFilePath = config['mapping_file_path']
        with open(mappingFilePath, 'r') as file:
            self.mapping = json.load(file)
            
        sequence_length = config['sequence_length']
        self.startSymbols = ["/"] * sequence_length
        
    def sampleWithTemperature(self, probabilities, temperature):
        predictions =  np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))     # Applying the softmax function
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

    """
    This function is used to generate the music. 
    Arguments:
    - seed: The seed or the input music to generate the next sequence.
    - numberOfSteps: The number of steps that the model needs to generate.
    - maxSequenceLength: The maximum length of the sequence that the model needs to consider to generate 
    """
    def generateMusic(self, seed, numberOfSteps, maxSequenceLength, temperature):
        # Create the seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self.startSymbols + seed
        
        # Map seed to integers
        seed = [self.mapping[symbol] for symbol in seed]
        
        for _ in range(numberOfSteps):
            # Limit the seed to the maximum sequence length
            seed = seed[-maxSequenceLength:]
            
            # One hot encode the seed
            oneHotEncodedSeed = keras.utils.to_categorical(seed, num_classes=len(self.mapping))
            oneHotEncodedSeed = oneHotEncodedSeed[np.newaxis, ...]     # Adding batch dimension so the model can process it
            
            # Make predictions
            probabilities = self.model.predict(oneHotEncodedSeed)[0]
            outputInteger = self.sampleWithTemperature(probabilities, temperature)
            
            # Update the seed
            seed.append(outputInteger)
            
            # Map the output integer back to the symbol
            outputSymbol = [symbol for symbol, index in self.mapping.items() if index == outputInteger][0]
            
            # Check if the output symbol is a stop symbol
            if outputSymbol == "/":
                break
            
            # Update the melody
            melody.append(outputSymbol)
            
        return melody
            
        
        
if __name__ == "__main__":
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
    log = logging.getLogger(__name__)
    
    melodyGenerator = MusicGenerator(config['saveModelPath'] + "/model.h5")
    maxSequenceLength = config['sequence_length']
    temperature = config['temperature']
    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _"
    melody = melodyGenerator.generateMusic(seed, numberOfSteps=100, maxSequenceLength=maxSequenceLength, temperature=temperature)
    print(melody)
    
    