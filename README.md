# NLP-Classification-3
Fake News Detection Using LSTM

ABOUT THE PROJECT:

Problem statement:
In the last few years, due to the widespread usage of online social networks, fake news spreading at an alarming rate for various commercial and political purposes which is a matter of concern as it has numerous psychological effects on offline society. According to Gartner Research

By 2022, most people in mature economies will consume more false information than true information.

For enterprises, this accelerated rate of fake news content on social media presents a challenging task to not only monitor closely what is being said about their brands directly but also in what contexts, because fake news will directly affect their brand value.


So, in this project, we will discuss how we can detect fake news accurately by analyzing news articles.

About Data:
The data used in this case study is the ISOT Fake News Dataset. The dataset contains two types of articles fake and real news. This dataset was collected from real-world sources; the truthful articles were obtained by crawling articles from Reuters.com (a news website). As for the fake news articles, they were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics, however, the majority of articles focused on political and World news topics.

The dataset consists of two CSV files. The first file named True.csv contains more than 12,600 articles from Reuters.com. The second file named Fake.csv contains more than 12,600 articles from different fake news outlet resources. Each article contains the following information:

article title (News Headline),
text (News Body),
subject
date
The total records in the dataset consist of 44898 records out of which 21417 are true news and 23481 are fake news.

Data processing:
In this step, we will read both the datasets (Fake.csv, True.csv) perform some data cleaning, merge both the datasets and shuffle the final dataset.
As we only need title and text, so we will drop extra features i.e., subject and date.
Similarly, we will be processing the True dataset and further merging it with the df_fake data frame to create the final dataset.
So, we have merged the dataset, but the dataset has a defined sequence of True and Fake labels. So, therefore we need to shuffle the dataset to introduce a randomness in the dataset.
From the dataset, we can observe that True news text contains the source of the news also i.e., 3rd row of True News text starts from BERLIN (Reuters) –, whereas 4th row of True News text starts from (Reuters) –

So, we have to clean the True News text as it affects the model building process as it’s not an important feature for building a fake news detection model.

Therefore, we have to write look ahead regular expression to retain only the text followed by “(Reuters) –“. 


Data preparation & cleaning:
Firstly we will convert the target variable label into binary variable 0 for True news and 1 for Fake news.
As we have to analyze the whole news article so we have to combine both title and text_processed features. Further, we have to drop unnecessary features i.e., title, text, and text_processed features from the dataset as we have to use only the final combined news column for building the model.
In the next step, lowercase the data, although it is commonly overlooked, it is one of the most effective techniques when the data is small. Although the word ‘Good’, ‘good’ and ‘GOOD’ are the same but the neural net model will assign different weights to it resulting in abrupt output which will affect the overall performance of the model.

Further, we will remove the stopwords and all non-alphabetic characters from the dataset.

Building Word Embeddings (GLOVE):
In this project, we will use Global vectors for word representation (GLOVE) word embeddings. It is an unsupervised learning algorithm for obtaining vector representations for words.
we have used a maximum sequence length of 100, a maximum vocab size of 20000, number of dimensions in this embedding is 50 i.e., each word has 50 dimensions in vector space, validation split of 0.2 will be used means 20% of the training data used for validating the model during the training phase. The batch size used here is 32. we have used  10 epochs to train the LSTM model.

Next, we will load the pre-trained word vectors from the embedding file.

Tokenize Text:

First, we have tokenized the final_news text using a tokenizer with a number of words equivalent to the maximum vocab size we have set earlier i.e., 20000.
The tokenizer.fit_on_texts is used to create a vocabulary index based on its frequency as it creates the vocabulary index based on word frequency, while tokenizer.text_to_sequences basically assign each text in a sentence into a sequence of integers. 
So  it takes each word in the sentence and replaces it with its corresponding integer value from word_index.

Sequence Padding:
Next, we have padded the sequence to the max length of 100 which we declared earlier. This is done to ensure that all the sequences are of the same length as is needed in the case of building a neural network.

Therefore, sequences which are shorter than the max length of 100 are padded with zeroes while longer sequences are truncated to a max length of 100.
Now we have assigned this sequence padded matrix as our feature vector X and our target variable y is label i.e., df[‘label’].
Then we have saved the word to id (integers) mapping obtained from the tokenizer into a new variable named word2idx as shown below.

Preparation of Embedding Matrix:
num_words are the minimum of Max_VOCAB_SIZE and length of word2idx+1.

We know MAX_VOCAB_SIZE = 20000 and length of word2idx = 29101. So the number of words is the minimum of these two i.e., 20000.

Next, an embedding matrix is created with the dimension of 50 and 20000 words. The words which will not be found in the matrix will be assigned as zeroes.

Creation of Embedding Layer:
Next, we will create an embedding layer that will be used as input in the LSTM model.

Model Building:
In this step, we will build the deep learning model Long Term Short Memory (LSTM). For this project, we will be using bi-directional LSTM.
we have used one hidden layer of the Bidirectional LSTM layer of 15 neurons. For accessing the hidden state output for each input time step we have to set return_sequences=”True”.

Model Parameters:

Here, the main thing to notice is that the total number of model parameters is 1,007,951 whereas training parameters are only 7951 which is the sum of parameters of bidirectional LSTM i.e., 7920 and dense layers parameter i.e., 31. The reason for this is that we have already set trainable = false in case of embedding layer due to which 1000000 parameters of embedding layers left non-trainable.

Train test split:
Now we split the feature and target variable into train and test sets in the ratio of 80:20, where 80% of the data is used for training the model and 20% is used for testing.

Fitting model:

In this step, we actually fit the model on the training set with batch size =32 and a validation split of 20%.
As we can observe from the above result, the model achieved a validation accuracy of 98.77% in just 10 epochs.

The training and validation loss and accuracy plots are shown below to show the progress of model training.
As we can see from the above plot, the model’s validation accuracy reached to 98% at around the 10th epoch.

Model result:
The training and test accuracy of the model is 98.73043894767761 and 98.017817735672.
As we can see from the above confusion matrix, the model has shown impressive performance. Now, we can have a look at the classification report to understand the overall performance from the statistical point of view.

Model Prediction:
This is the most important part of this project i.e., actually using the model to detect fake and real news from sample news articles. But for that, we have to perform certain preprocessing of sample text before feeding them into the LSTM model.

Conclusion:
In this project, we have demonstrated the application of the deep learning model LSTM for the detection of fake news from news articles and it showed outstanding results. Further, we can utilize this model for verifying the genuineness of the online news article generally cropped up on social media websites by integrating this model in the form of a Google Chrome extension or separate web application.















