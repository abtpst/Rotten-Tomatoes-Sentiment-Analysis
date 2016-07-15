### Solving sentiment analysis with 3 different classifiers

In this package, we are going to perform sentiment analysis using the following machine learning techniques

1. Logistic Regression
2. Multinomial Naive Bayes
3. Stochastic Gradient Descent

### Approach

In this problem we are dealing with five different values of classification. Since this is not binary, there is bound to be some overlap in the feature vectors of the classes. For instance, 'Positive' and 'Somewhat Positive' sentences are likely to contain similar words.

We are going to create five different binary classifiers, one for each sentiment. For example, we can create a binary classifier for sentences that are 'Positive' and sentences that are not. Similarly for the other sentiment values. Then, finally, we will combine the results.

### Steps

#### 1. Preprocessing
First, we will create five sets of training data from our original training set. These will be fed to our binary classifiers. Each of these sets can be taken as training data for one value of the sentiment. For example, lets say we mark all sentences with a 'Positive' sentiment as 1 and all other as 0 and we call this marked data as `train0`. Now, `train0` can be used to train a binary classifier to predict whether a given sentence falls under 'Positive' or not. We can do similar things for the other sentiment values. This is achieved by the `prepData.py` script.

#### 2. Build term document matrix using TfidfVectorizer 
Next, we create the term document matrices for each of the five sets of training data we created. We also create term document matrices for the test set. Before feeding the data to TfidfVectorizer, we must clean it up to remove anything that we think will not add value. For example, special symbols or numbers might not be as important. This cleanup is handled by the `preProc.py` module.