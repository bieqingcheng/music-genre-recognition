# music-genre-recognition
Multi-level local feature coding fusion for music genre recognition
## Extract features
a) In this work, we used Marsyas toolbox to generate the timbre features. For documentation, see the url: http://marsyas.info/ <br>
b) In this work, we used the TU Wien's implementation to generate the rhythm histogram (RH) and statistical spectrum descriptor (SSD).
   (https://github.com/tuwien-musicir/rp_extract) <br>
c) In this work, we used the transfer feature proposed by Keunwoo Choi. (https://github.com/keunwoochoi/transfer_learning_music) <br>
d) For Mel-spectrogram, constant-Q spectrogram, harmonic spectrogram and percussive spectrogram, we used the Librosa 
implementation. For scatter transform spectrogram, we used the Kymatio implementation. More detail about these five features 
can be seen in the folder extreact_features.

## Dataset split
Split the dataset and store each fold of data in a txt file.
usage:
``` python
python fold_10_split.py
```
