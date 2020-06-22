# music-genre-recognition
Multi-level local feature coding fusion for music genre recognition
## Extract features
a) In this work, we used Marsyas toolbox to generate the timbre features. For documentation, see the url: http://marsyas.info/ <br>
b) In this work, we used the TU Wien's implementation to generate the rhythm histogram (RH) and statistical spectrum descriptor (SSD).
   (https://github.com/tuwien-musicir/rp_extract) <br>
c) In this work, we used the transfer feature proposed by Keunwoo Choi. (https://github.com/keunwoochoi/transfer_learning_music) <br>
d) For Mel-spectrogram, constant-Q spectrogram, harmonic spectrogram and percussive spectrogram, we used the Librosa 
implementation. For scatter transform spectrogram, we used the Kymatio implementation. More detail about these five features 
can be seen in the folder `extreact_features`.

## Dataset split
Split the dataset and store each fold of data in a txt file.<br>

``` python
python fold_10_split.py
```
Please pay attention to modify the data input and output path.
## Train a feature encoding network for each feature and extract the feature coding.
The extraction of feature codes includes three stages: feature network data reading, feature code network construction, feature
code network training set feature code extraction. The feature encoding network involves numerous parameters and file paths, 
so first define a configuration file to facilitate subsequent data reading and network training.
'config.py' is a configuration file for all the feature encoding networks.<br>
`For each feature`:<br>
The python file starting with "dataloader" is a data reading file.<br>
The python file starting with "model" is a model file.<br>
The python file starting with "main" is a main file.<br>
Take Mel-spectrogram as an example to introduce model training, and the usage 
of the model corresponding to other features is the same.<br>
`training:`<br>
```python
python main_mel_6_14.py
```
When the training process is finished, evaluate the model using the testing dataset. And extract the feature coding for training 
dataset and testing dataset in each fold, respectively.
```python
python main_mel_6_14.py --resume 12 -e
```
Please note that lines 98, 301, 313, 317 and 318 may need to be modified when 
extracting the feature encoding of the training dataset and testing dataset.

## Use the sum rule to initially select the top 6 feature combinations that yields the highest testing accuracies.
```python
python combination_6_14.py
```
## Final decision
After selecting the top 6 feature combinations that yields the highest testing accuracies, a meta CNN is used to 
learn aggregated high level features for final genre classification based on these top 6 feature combinations. <br>

`training:`<br> 
```python
python main_fusion_sT_6_14.py
```
`evaluating:` <br>
```python
python main_fusion_sT_6_14.py --reusme 12 -e
```
## Calculate the confusion matrix with the effective measures.
```python
python matrix_confusion.py
```
