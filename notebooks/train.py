# Imports
import yaml
import numpy as np
from DataPreprocessing import generating_training_sequences
import logging
import logging.config

def main():
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
    log = logging.getLogger(__name__)
    
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


if __name__ == "__main__":
    main()