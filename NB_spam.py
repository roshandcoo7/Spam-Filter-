import os #to interface with operating system
import _pickle
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

def counts_of_words():
    direc = r"C:\Users\sahadudheen\Documents\Machine Learning\mails"

    files = os.listdir(direc) #returns a list containing the names of entries in the given path

    emails = list() #list of all the mails
    for i in files:
        emails.append(direc + '\\' + i)

    words = list() #list all the words in the data set
    for mail in emails:
        f = open(mail,'r', encoding='latin1').read()
        words += f.split(" ")

    fin_W = list() #final list of words
    for i in range(len(words)):
        if words[i].isalpha() == True: #to remove all the special characters like ,., ,\,?,!,@.....
            fin_W.append(words[i])

    dictionary = dict()
    dictionary = Counter(fin_W) #returns a dictioanry with the count of each of the elements in the input list 
    return dictionary.most_common(3000)

def make_dataset(dictionary):
    direc = r"C:\Users\sahadudheen\Documents\Machine Learning\mails"

    files = os.listdir(direc) #returns a list containing the names of entries in the given path

    emails = list() #list of all the mails
    for i in files:
        emails.append(direc + '\\' + i)
    
    c = len(emails)
    feature_vector = list()
    labels = list()
    for mail in emails:

        data = list()
        f = open(mail,'r', encoding='latin1').read()
        words = f.split(" ") #words in an individual email

        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_vector.append(data)

        if "ham" in mail:
            labels.append(0)
        if "spam" in mail:
            labels.append(1)
        print(c)
        c = c - 1
    
    labels.append(0)
    return feature_vector, labels

d = counts_of_words()
features, labels = make_dataset(d)
print(len(features),len(labels))
x_train,x_test,y_train,y_test = tts( features,labels, test_size = 0.2 )

clf = MultinomialNB()
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
print(accuracy_score(y_test,predictions))

filename = 'finalized_model.mdl'
_pickle.dump(clf, open(filename, 'wb'))
