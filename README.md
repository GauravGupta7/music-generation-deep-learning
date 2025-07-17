# Music-Generation-Using-LSTMs

### 🔶 **Foreground**

Any musical piece comprises of a characteristic sequence of musical notes (denoting different frequencies and pitches) played at different time instant. The think that needs to be noted here is that the notes that are being played cannot be played randomly. In layman's term, the adjacent notes should have some sort of coherence within them so that the tune sounds melodius. In this project, we will try to develop an RNN-LSTM based AI model to predict the next note in the series using the previous notes. For this we will try to utilize the coherence between adjacent notes.

### 🔶 **Brief on Data Used**
We are using the kern humdrum dataset to train the model. This data contains the folksongs from Germany. Here, **kern** is the core pitch and duration representation for common practice music notation. The kern representation allows the pitch and canonical duration information to be encoded.  
  
Three types of data tokens are distinguised in **kerns:** notes, rest and barlines. These three data are some of the basic properties that determine the tonal quality of a musical piece. **Notes** can encode a variety of attributes including absolute pitch, accidental, canonical duration, articulation, ornamentation, ties, slurs, phrasing, stem-direction and beaming. 

More about **kerns** can be studied from: https://www.humdrum.org/rep/kern/index.html  
The dataset can be downloaded from: https://kern.humdrum.org/cgi-bin/browse?l=essen/europa/deutschl

### 🔶 **Data Preprocessing**
The model that we are building needs the training the data to be in proper format for it to understand and gain insights from it. For this the data needs to be cleaned and encoded in suitable format. 

We are implementing the transposition of the song. A particular song can be of any of the 24 available scales. Thus, training our model on all the 24 different scales is unnecessary and inefficient. This will cause extreme overhead while training the model. So we implement transposition and shift the songs on major scales to C<sub>Maj</sub> and the songs on minor scales to A<sub>m</sub>.

Later, we are encoding the song to machine readable format. We are performing encoding on each and every song and then we are storing the output as '.txt' files in a separate folder. Later, for convenience while model training, we are merging the processed songs into a single file dataset. 

We are also using one-hot encoding in the data preprocessing stage so that we eliminate the ordinal relationships among the numbers. Since, we have created a categorical dataset during the preprocessing stage, it becomes extremely important for us to implement O.H.E for getting optimum performance from out model. 

Now, due to resource limitations, running preprocessing every time becomes difficult. So we are storing the inputs and targets of as separate standalone file in our repository so that we don't have to execute preprocessing every time. 

### 🔶 **Understanding LSTM**
LSTM stands for **Long Short Term Memory**. It is a type of RNN (Recurrent Neural Network) architecture designed to facilitate learning of long term dependencies in sequential data. The Recurrent or very deep neural networks are challenging to train as they suffer from exploding/vanishing gradient problem<sup>[2]</sup>. To overcome this when learning the long term dependencies, the LSTM architecture was introduced. The LSTM architecture implements this using special type of memory cells that act as a conveyer belt carrying information across multiple time units.

The LSTM architecture have been put to use in below mentioned scenarios:
* Natural Language processing
* Time Series Analysis
* Speech Recognition

LSTM can help the RRN model to selectively remember and forget information, making them and efficient strategy in capturing long term dependencies in sequential data that traditional RRNs. 

### 🔶 **Tools and Libraries Used**  
* **music21:** This is a python based toolkit for computer-aided musicology. It is an all rounder toolkit that can be used to generate, visualize and study music using python. The detailed documentation for music21 toolkit can be found at https://www.music21.org/music21docs/usersGuide/index.html  

### 🔶 **Important Configuration Parameters**

* Test and preprocessed data paths
```yaml
dataset_path: data/essen/europa/deutschl/test_dataset
save_path: data/essen/europa/deutschl/test_dataset
mapping_file_path: data/essen/europa/deutschl/mapping.json
```
* The song beats duration that will be accepted by the model can be passed inside this list

```yaml
acceptable_durations: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
```
* The length of the sequence that will be used to train the LSTM
```yaml
sequence_length: 64
```
* The amount of data that you want to use for preprocessing and training is restricted due to resources constraints on local PC.
```yaml
data_size: 150000
```
* If you want the preprocessing funtions to execute then the value of the below parameter can be updated accordingly
```yaml
isPreprocessingRequired: True 
```
* Since we are avoiding preprocessing when not needed, we are storing the preprocessed tensors at the below folder path, and loading them from here when needed
```yaml
testTargetPath: processed_tensors
```

### 🔶 **References**
[1] https://www.humdrum.org/rep/kern/index.html  
[2] https://www.researchgate.net/publication/340493274_A_Review_on_the_Long_Short-Term_Memory_Model