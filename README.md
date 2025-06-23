# Music-Generation-Using-LSTMs

## 🔴 **Foreground**

A musical piece comprises of a characteristic sequence of musical notes (denoting different frequencies and pitches) played at different time instant. The think that needs to be noted here is that the notes that are being played cannot be played randomly. In layman's term the adjacent notes should have some sort of coherence within them so that the tune sounds melodius. In this project, we will try to develop some sort of mechanism that can predict the next note in the series using the previous notes. For this we will try to utilize the coherence between adjacent notes.

## 🔴 **Brief on Data Used**
We are using the kern humdrum dataset to train the model. This data contains the folksongs from Germany. Here, **kern** is the core pitch and duration representation for common practice music notation. The kern representation allows the pitch and canonical duration information to be encoded.  
  
Three types of data tokens are distinguised in **kerns:** notes, rest and barlines. These three data are some of the basic properties that determine the tonal quality of a musical piece. **Notes** can encode a variety of attributes including absolute pitch, accidental, canonical duration, articulation, ornamentation, ties, slurs, phrasing, stem-direction and beaming. 

More about **kerns** can be studied from: https://www.humdrum.org/rep/kern/index.html  
The dataset can be downloaded from: https://kern.humdrum.org/cgi-bin/browse?l=essen/europa/deutschl

## 🔴 **Data Preprocessing**
The model that we are building needs the training the data to be in proper format for it to understand and gain insights from it. For this the data needs to be cleaned and encoded in suitable format. 

We are implementing the transposition of the song. A particular song can be of any of the 24 available scales. Thus, training our model on all the 24 different scales is unnecessary and inefficient. This will cause extreme overhead while training the model. So we implement transposition and shift the songs on major scales to C<sub>Maj</sub> and the songs on minor scales to A<sub>m</sub>.

## 🔴 **Understanding LSTM**
LSTM stands for **Long Short Term Memory**. It is a type of RNN (Recurrent Neural Network) architecture designed to facilitate learning of long term dependencies in sequential data. The Recurrent or very deep neural networks are challenging to train as they suffer from exploding/vanishing gradient problem<sup>[2]</sup>. To overcome this when learning the long term dependencies, the LSTM architecture was introduced. The LSTM architecture implements this using special type of memory cells that act as a conveyer belt carrying information across multiple time units.

The LSTM architecture have been put to use in below mentioned scenarios:
* Natural Language processing
* Time Series Analysis
* Speech Recognition

LSTM can help the RRN model to selectively remember and forget information, making them and efficient strategy in capturing long term dependencies in sequential data that traditional RRNs. 

## 🔴 **Tools and Libraries Used**
* **music21:** This is a python based toolkit for computer-aided musicology. It is an all rounder toolkit that can be used to generate, visualize and study music using python. The detailed documentation for music21 toolkit can be found at https://www.music21.org/music21docs/usersGuide/index.html

## 🔴 **References**
[1] https://www.humdrum.org/rep/kern/index.html  
[2] https://www.researchgate.net/publication/340493274_A_Review_on_the_Long_Short-Term_Memory_Model