# Imports
import yaml
import numpy as np

if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
    
from DataPreprocessing import generating_training_sequences
import logging
import logging.config
import tensorflow.keras as keras


def buildModel(outputUnits, numberOfUnits, lossFunction, learningRate):
    log.info("Building the model architecture...")
    # Create the model arcitecture
    inputLayer = keras.layers.Input(shape=(None, outputUnits))
    x = keras.layers.LSTM(numberOfUnits[0])(inputLayer)
    x = keras.layers.Dropout(0.2)(x)  # Adding dropout to prevent overfitting
    outputLayer = keras.layers.Dense(outputUnits, activation='softmax')(x)  # Output layer with softmax activation
    model = keras.Model(inputs=inputLayer, outputs=outputLayer)
    
    # Compiling the model
    log.info("Compiling the model...")
    model.compile(loss = lossFunction,
                  optimizer=keras.optimizers.Adam(learning_rate=learningRate),
                  metrics=['accuracy'])
    model.summary()
    
    # Returning the model
    log.info("Model architecture built successfully.")
    return model
    


def main():
    # Loading configuration parameters
    sequence_length = config['sequence_length']
    save_path = config['save_path']
    mapping_file_path = config['mapping_file_path']
    training_data_size = config.get('data_size', 50000)  # Default to 50000 if not specified
    isPreprocess = config.get('isPreprocessingRequired', True)  # Default to True if not specified
    testTargetFolderPath = config['testTargetPath']
    
    if isPreprocess:
        log.info("Starting preprocessing...")
        inputs, targets = generating_training_sequences(sequence_length, save_path, mapping_file_path, training_data_size)
        np.save(testTargetFolderPath+"/inputs.npy", inputs)               # Saving the inputs to avoid preprocessing every time
        np.save(testTargetFolderPath+"/targets.npy", targets)             # Saving the targets to avoid preprocessing every time
    else:
        log.info("Skipping preprocessing, loading preprocessed data...")
        inputs = np.load(testTargetFolderPath+"/inputs.npy")              # Loading the inputs from the saved file
        targets = np.load(testTargetFolderPath+"/inputs.npy")            # Loading the targets from the saved file
        log.debug(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    log.info("Execution completed successfully...")
    
    # Build the model
    outputUnits = 22  # Number of unique symbols in the dataset
    lossFunction = config['lossFunction']
    learningRate = config['learningRate']
    numberOfUnits = config['numberOfUnits']
    model = buildModel(outputUnits, numberOfUnits, lossFunction, learningRate)
    
    #Training the model
    log.info("Starting model training...")
    numberOfEpocs = config['numberOfEpocs']
    trainingBatchSize = config['trainingBatchSize']
    log.info("Inputs shape: %s", inputs.shape)
    num_sequences = training_data_size//sequence_length
    inputs = inputs[:num_sequences * sequence_length]
    inputs = inputs.reshape((num_sequences, sequence_length, 22))
    model.fit(inputs, 
              targets, 
              epochs=numberOfEpocs, 
              batch_size=trainingBatchSize, 
              validation_split=0.2)
    log.info("Model training completed successfully!")
    
    # Saving the model
    log.info("Saving the model...")
    saveModelPath = config['saveModelPath'] + "/model.h5"
    model.save(saveModelPath)
    log.info(f"Model saved successfully at {saveModelPath}")


if __name__ == "__main__":
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
    log = logging.getLogger(__name__)
    main()