import os 
import io 
import numpy
import random
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer


########################################################################################################################

data = DataFrame({'message': [], 'class': []}) #makes a dataframe of two lists,one with mail and it's type(spam/ham)

data = data.append(dataFrameFromDirectory('C:\MLCourse\emails\spam', 'spam')) #give the location of where these spam mails are
data = data.append(dataFrameFromDirectory('C:\MLCourse\emails\ham', 'ham')) #give the location of where these spam mails are
data = data.reindex(numpy.random.permutation(data.index)) #shuffle the dataframe

print(data)

traindata = data[:2400]
testdata = data['message'][2400:]

#############################DOING NAIVE BAYERS ON THE DATA WE HAVE EXTRACTED##########################################

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(traindata['message'].values) #converts each word into a number and counts how many times it has occured

classifier = MultinomialNB()
targets = traindata['class'].values
classifier.fit(counts, targets)

############################NOW TEST WITH TEST TW0 MAILS###############################################################

example_counts = vectorizer.transform(testdata)
predictions = classifier.predict(example_counts)


##########################################FIND ACCURACY#############################################################
print(predictions) 


    

