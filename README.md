# birds_dnn
Feature extraction (frontend) and machine learning for classification of birds songs using Kaggle dataset.

# Install

After using git clone, copy the folder with ogg files to a folder called train_audio, and locate  train_audio in the same folder as src, etc.

For instance, if you cloned the repository to C:\github\birds_dnn\ and now have
```
C:\github\birds_dnn\src
C:\github\birds_dnn\txt
```
then, create the folder with your ogg files:
```
C:\github\birds_dnn\train_audio
```

# Executing code

From src folder, you can use:
```
 python .\listen_and_see.py --should_plot --features=stft --normalization_method=minmax
 ```
 to listen to the files and also plot images with spectrogram or similar features.

# To do list

1) Passar os datasets de audio (waveform) que tás usando (treino e teste)

2) Passar os scripts que geram espectrogramas

3) Me ensinar a rodar script nos datasets de forma a gerar exatamente os dados do teu Colab 



4) Só há um bird por file? Só um label por file?

5) Can we use metadata (lat / long) or only the audio waveform?

6) about the submission file: for each row, we print the 182 values of the output softmax layer?

7) change the Colab notebook to read zip files from a LASSE nextcloud folder?

# References

https://www.sciencedirect.com/science/article/abs/pii/S0003682X2300631X

https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14239

https://www.hindawi.com/journals/mpe/2021/8872248/

https://arxiv.org/pdf/2202.02112
