import numpy as np
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict  # Fix for compatibility issues

import logging
import logging.config
import tensorflow.keras as keras
import yaml
import json

class MusicGenerator:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.model = keras.models.load_model(modelPath)

        # Load the mapping file
        mappingFilePath = config['mapping_file_path']
        with open(mappingFilePath, 'r') as file:
            self.mapping = json.load(file)
        
        # Reverse mapping for integer to symbol lookup
        self.reverseMapping = {v: k for k, v in self.mapping.items()}

        sequence_length = config['sequence_length']
        self.startSymbols = ["/"] * sequence_length

    def sampleWithTemperature(self, probabilities, temperature):
        predictions = np.log(probabilities + 1e-9) / temperature  # Added epsilon for numerical stability
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        return np.random.choice(choices, p=probabilities)

    def generateMusic(self, seed, numberOfSteps, maxSequenceLength, temperature):
        seed = seed.split()
        melody = seed.copy()
        seed = self.startSymbols + seed

        try:
            seed = [self.mapping[symbol] for symbol in seed]
        except KeyError as e:
            logging.error(f"Symbol '{e.args[0]}' not found in mapping. Please check your seed or mapping file.")
            return []

        for _ in range(numberOfSteps):
            seed = seed[-maxSequenceLength:]

            oneHotEncodedSeed = keras.utils.to_categorical(seed, num_classes=len(self.mapping))
            oneHotEncodedSeed = oneHotEncodedSeed[np.newaxis, ...]

            probabilities = self.model.predict(oneHotEncodedSeed, verbose=0)[0]
            outputInteger = self.sampleWithTemperature(probabilities, temperature)

            # Get symbol from reverse mapping safely
            outputSymbol = self.reverseMapping.get(outputInteger)
            if outputSymbol is None:
                logging.warning(f"Predicted index {outputInteger} not in mapping. Skipping step.")
                continue

            if outputSymbol == "/":
                break

            seed.append(outputInteger)
            melody.append(outputSymbol)

        return melody


if __name__ == "__main__":
    # Load logging and application config
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    melodyGenerator = MusicGenerator(config['saveModelPath'] + "/model.h5")
    maxSequenceLength = config['sequence_length']
    temperature = config['temperature']

    seed = "55 _ _ _ 60 _ _ _ 55 _ _ _ 55 _ _ _"
    melody = melodyGenerator.generateMusic(
        seed,
        numberOfSteps=100,
        maxSequenceLength=maxSequenceLength,
        temperature=temperature
    )

    print("Generated Melody:")
    print(" ".join(melody))
