# Imports
import yaml
from DataPreprocessing import generating_training_sequences

def main():
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    sequence_length = config['sequence_length']
    save_path = config['save_path']
    mapping_file_path = config['mapping_file_path']
    training_data_size = config.get('data_size', 50000)  # Default to 50000 if not specified
    inputs, targets = generating_training_sequences(sequence_length, save_path, mapping_file_path, training_data_size)
    print("Inputs shape:", inputs.shape)
    print("Targets shape:", targets.shape)
    print("Execution completed successfully...")


if __name__ == "__main__":
    main()