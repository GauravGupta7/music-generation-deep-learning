# Importing libraries
import numpy as np
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict  # Fix for compatibility issues

import os
import music21 as m21
import json
import tensorflow.keras as keras
import numpy as np
import yaml

# defining a funtion to load all the files of the data set
songs = []

def load_songs_kern_dataset(dataset_path):
    for path, subdirs, files in os.walk(dataset_path):
        
        # we need to filter out the ".krn" files from the dataset
        for file in files:
            if file.endswith('.krn'):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


# Function to transpose a song to another scale
def transpose(song): 
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measure_part0[0][4]
    
    # if key not present then estimate the key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze('key')
        
    # get the interval for transposition (example: BMaj to CMaj)
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))
        
    # transpose the song
    transposed_song = song.transpose(interval)
    
    return transposed_song  


# Function to encode pitch and duration of song to machine-readable format
def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'
            
        # convert the notes and rests into time series notation
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')
                
    # Calculate the duration of the song
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


# function to preproces the songs dataset and prepare the data for our model
def preprocess_songs(dataset_path, save_path, acceptable_durations):
    # Load the songs from the dataset
    songs = load_songs_kern_dataset(dataset_path)
    
    for i, song in enumerate(songs):
        # Filter songs based on acceptable durations
        if not has_acceptable_duration(song, acceptable_durations):
            continue
    
        # Transpose song to C major or A minor
        song = transpose(song)
        
        # Encode songs with music time series representation
        encoded_song = encode_song(song)   
        
        # Save the encoded song to a file in save path
        saved_path = os.path.join(save_path,  f"song_{i}.txt")
        with open(saved_path, 'w') as fp:
            fp.write(encoded_song)
            

# Function to check whether a song has acceptable duration
def has_acceptable_duration(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def load(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song


# Creating a single file that will contain all the songs
def create_single_file_dataset(dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    
    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    
    songs = songs[:-1]     # remove the last delimiter to avoid trailing spaces
            
    # save the string in the single file
    #dataset_path1 = os.path.join(dataset_path,  f"songs.txt")
    dataset_path1 = dataset_path + "/songs.txt"
    os.makedirs(os.path.dirname(dataset_path1), exist_ok=True)
    with open(dataset_path1, "w") as fp:
        fp.write(songs)
        
    return songs


# Creating a lookup table for all the symbols used in the final dataset
def create_mapping(songs, mapping_file_path):
    mappings = {}
     
    # Identify the vocabluary
    songs = songs.split()
    vocabulary = list(set(songs))
     
    # Create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
         
    # save the vocalbuary to a file
    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
    with open(mapping_file_path, 'w') as fp:
        json.dump(mappings, fp, indent=4)
        
        
def convert_songs_to_int(songs, mapping_file_path):
    int_songs = []
    
    # Load mappings
    with open(mapping_file_path, 'r') as fp:
        mappings = json.load(fp)
        
    # Cast song string to a list
    songs = songs.split()
    
    # Map song to the int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs


# Function to generate training seeqe
def generating_training_sequences(sequence_length, save_path, mapping_file_path, training_data_size=50000):
    # Load songs and map them to integers
    single_file_dataset = save_path + "/songs.txt"
    songs = load(single_file_dataset)
    int_songs = convert_songs_to_int(songs, mapping_file_path)
    
    # Generate training sequences
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs = np.append(inputs, int_songs[i:i + sequence_length])
        targets = np.append(targets, int_songs[i + sequence_length])
        inputs = inputs[:training_data_size]  # Limit the number of inputs
        targets = targets[:training_data_size]  # Limit the number of targets
        
        # One hot encoding the sequence
        vocabulary_size = len(set(int_songs))
        inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
        targets = np.array(targets)
        
    return inputs, targets



def main():
    with open("config/application.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    dataset_path = config['dataset_path']
    acceptable_durations = config['acceptable_durations']
    save_path = config['save_path']
    sequence_length = config['sequence_length']
    mapping_file_path = config['mapping_file_path']
    training_data_size = config.get('data_size', 50000)  # Default to 50000 if not specified
    preprocess_songs(dataset_path, save_path, acceptable_durations)
    songs = create_single_file_dataset(save_path, sequence_length)
    create_mapping(songs, mapping_file_path)
    inputs, targets = generating_training_sequences(sequence_length, save_path, mapping_file_path, training_data_size)
    print("Execution completed successfully...")
    

if __name__ == "__main__":
    main()