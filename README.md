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

 One can find more information in the file processing_steps.docx located in folder documentation.

# About the Kaggle challenge

- One may have more than a bird singing in one file. But we don't know how the label file describes it

- In the Kaggle challenge, one can use metadata (example: lat / long)

- About the submission file: for each row, we print the 182 values of the output softmax layer

# References

https://www.sciencedirect.com/science/article/abs/pii/S0003682X2300631X

https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14239

https://www.hindawi.com/journals/mpe/2021/8872248/

https://arxiv.org/pdf/2202.02112
