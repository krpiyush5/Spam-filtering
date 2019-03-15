# Spam-filtering


SPAM FILTERING:-

Spam Filter is a machine learning model that is used to detect unsolicited and unwanted messages and avert those messages from getting to a users’s inbox.

We will use  sms spam data from kaggle link given below.

#https://www.kaggle.com/uciml/sms-spam-collection-dataset

Algorithm:-

The most challenging task in machine learning is to apply algorithm because we can not apply any random algorithm to any random data.

It’s all about visualisation and preception of data ,first of all it is a classification model because here class values are only two spam or ham .We have two columns one is text or messages and other one is class spam or ham.There are  some classification techniques like knn,naive bayes ,logistic regression ,but after thinking intuitively we will use Naive Bayes Algorithm  because it uses Bayes theorem to determine the probability that a give message is spam ,given words in this message.

If S is the event of a given message being spam and w is a word in the message,we will classify it as spam with probability:

P(S | w) =P(w | S)*P(S)  /  { P(w|S) *P(S) +P(w|S’)*P(S’) }
Preprocessing and vectorisation:-

First of all we need to preprocess our text data and convert it into vectors with the help of featurisation like BOW(Bag of Words) ,TFIDF,Word2vec,Avg TFIDF Word2vec.Naive Bayes uses probability of data so the data values should be positive but in the case of Word2vec and Avg TFIDF Word2vec vectors may be postive or negative so we can’t use this featurisation.

After preprocessing we will split our data into train and test then convert into vectors with the help of BOW and TFIDF.

We will use Scikit library of python to implement Naive Bayes because of it’s lucidity.

NOTE:-

Laplace Smoothing:-  It is used to smooth categorical data.Lets take an observation
x=(x1,x2,x3........,xd) from a multinomial distribution with N trials  then,

our estimator=(x+alpha) / (N+alpha*d) .

High bias will lead to underfitting and high variance will lead to overfitting.

So we need to figure out our best alpha value such that our area under ROC curve maximize.

Then we will fit our train data for optimal alpha and predict for test data and summarise it into confusion matrix using heatmaps.
