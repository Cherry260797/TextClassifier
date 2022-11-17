# TextClassifier
# Naive Bayes Classifier
Naive Bayes Classifier works as the probabilistic classifier drawing it's implications from Bayes Theorem of Probability Theory and Naive assumption. [1] Naive Bayes Algorithm is a supervised learning algorithm as the labels as the ground truths are available with the training data. 

# Probability 
 It's the extent to which an event is likely to occur. It's measured as the ratio of favorable cases to the total number of cases. Probabilistic way of thinking can be classified as Bayesian thinking and Frequentist thinking. Frequentist thinking considers only evidence where as Bayesian perspective considers both prior belief and evidence.

# Conditional Probability(P(A/B)) : 
Consider A and B are two events, probability of event A occuring given that event B occured or event B is true. It's the posterior probability of A given B.

                                                  P(A/B)=P(A,B)/P(B) 

Here P(A,B) is Joint Probability which represents probability of both events A and B occuring.

P(A/B)=P(A,B)/P(B)  and P(B/A)=P(A,B)/P(A)

Joint Probability is symmetrical=>P(A/B)*P(B)=P(B/A)*P(A)

P(A/B)=P(B/A)*P(A)/P(B)

# Bayes Theorem 
It provides basis for probabilistic learning that addresses the observed data along with the prior knowledge.

P(A/B)=P(B/A)*P(A)/P(B)

# Independent Events  
If occurence of A doesn't effect the occurence of B then events A and B are said to be Independent events. When events A and B are independent, their joint probability is given by

P(A,B)=P(A)*P(B)

# Naive Assumption 
For implementing the Naive Bayes Classifier we consider Naive Assumption that states events A and B are independent provided C. This is Conditional Independence.

# Conditional Probability and Laplace Smoothing
Consider the problem containing attribute set X, and class label C then for a data instance with set of attribute values x, probability of observing each class label c is calculated and the maximum probability giving class is predicted.
 In case of small training instances, large possible attribute values, missing combination of attribute values class label results in zero conditional probability. It's impossible to classify if one of the test attributes has a data value that is missing in the training data points. This is given by Laplace estimate or Laplace smoothing.
P(x1=a|c)=nc+1/n+v
Here c is class label, nc is number of training data points (with xi=a) of given class, v is number of features, n is number of training instances with xi=a.

# Text Classifier Problem
# Ford Sentence Classification DataSet 
This dataset has the sentences that resemble the job descriptions and a class label that represents the class of the sentence. For example a sentence might be related to Responsibility or Experience or Skill or Soft Skills or Education or Requirement. Here there are train_csv file has 60,116 entries mentioning the sentences under the attribute 'New_Sentence' and the class label they belong to under the attribute 'Type.' In all, the classification has to be done among the six classes of Responsibility, Requirement, Skill, Softskill, Experience and Education. Now using this training data, a Machine Learning model is to be built in order to classify the given job description sentence into one of the above six classes.

I've built a Naive Bayes Classifier from scratch for this classification and here is the procedure walk through describing it.

# Step 1 :
#  Importing the required packages :

I've created a python notebook and imported the packages of pandas, nltk.

Pandas is to read the contents of the csv files into dataframes.

nltk is to utilize the functionalities related to stop words. Stop words are the most common words of the vocabulary that do not influence the classification but occur more frequently.

import nltk
from nltk.corpus import stopwords
import pandas as pd

# Step 2 :
# Pre-Processing Data :

# Reading Data from CSV file :

pd.read_csv function with the input of file path reads the file contents into data frame. Here train_csv contents of 60K rows is read into data frame named 'data.'


# Pre-Process Data :

There are entries in the training data that doesn't contain contents in New_Sentence attribute that is they have null values. These entries can be dropped as they are of no use.

Dataframe. drpona() function can be used to drop all the null values attribute rows.


# Removal of Stop Words :

Using the nltk package's, they are removed from the 'New_Sentence' attribute's contents from all the rows using the code below

stop_words = set(stopwords.words('english'))
data['New_Sentence'] = data['New_Sentence'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
# Convert the Data Frame Contents into List :

Convert the contents of the data frame 'data' into list of lists using values.tolist() function.


# Retrieve Attribute Set and Labels into separate lists :

All the rows or sentences under the 'New_Sentence' attribute are retrieved into a list named 'datat' and all their pertaining class labels are retrieved into a list named 'target' using the code mentioned below.

Looping through the length of the data_new list, all contents of 'New_Sentence' are appended to 'datat' list and depending upon the value of the 'Type', a numerical value ( 0 for Education, 1 for Experience, 2 for Requirement, 3 for Responsibility, 4 for Skill, 5 for Soft Skill) is appended to the target list.

for i in range(len(data_new)):
    datat.append(data_new[i][2])
    if (data_new[i][3]=='Experience'):
        target.append('1')
    if (data_new[i][3]=='Responsibility'):
        target.append('3')
    if (data_new[i][3]=='Requirement'):
        target.append('2')
    if (data_new[i][3]=='SoftSkill'):
        target.append('5')
    if (data_new[i][3]=='Skill'):
        target.append('4')
    if (data_new[i][3]=='Education'):
        target.append('0')
Performing the Train, Dev and Test Split :

60%, 20%, 20% split is done for training, development and testing. Sizes of training, development and test sets are calculated based on size of the dataset 'datat'

train_size=60/100*len(datat)
dev_size=80/100*len(datat)
test_size=math.ceil(len(datat))
# Step 3 :
# Building The Model :

# Create dictionaries to save the word count :

Create dictionary to save the word count of all sentences in dictionary named 'counts.'

Create 6 dictionaries to save word counts of each class belonging sentences into pertaining dictionary.

counts=dict()
zero_counts=dict()
one_counts=dict()
two_counts=dict()
three_counts=dict()
four_counts=dict()
five_counts=dict()
# Count the words of all sentences into dictionary counts :

If word already exists in the dictionary, count is incremented and otherwise, word is added to the dictionary with count 1.

for i in range(math.ceil(train_size)):
 str=datat[i]
 #print(str,i)
 words = str.split(' ')
 for word in words:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1
# Count the occurance of each class label to find the prior class probabilities :

From the list named target, occurances of each class are counted in order to find the prior probabilities.

c0=c1=c2=c3=c4=c5=0
for i in range(math.ceil(train_size)):
    if(target[i]=='0'):
        c0+=1
    if(target[i]=='1'):
        c1+=1
    if(target[i]=='2'):
        c2+=1
    if(target[i]=='3'):
        c3+=1
    if(target[i]=='4'):
        c4+=1
    if(target[i]=='5'):
        c5+=1
# Count the word counts of each class into pertaining dictionaries :

Word counts of sentences belonging to each class are required to calculate the conditional class probabilities. Therefore, they are calculated.

for i in range(math.ceil(train_size)):
    str=datat[i]
    words = str.split(' ')
    if(target[i]=='0'):
     for word in words:
        if word in zero_counts:
            zero_counts[word] += 1
        else:
            zero_counts[word] = 1
    if(target[i]=='1'):
        for word in words:
            if word in one_counts:
                one_counts[word] += 1
            else:
                one_counts[word] = 1
    if(target[i]=='2'):
        for word in words:
            if word in two_counts:
                two_counts[word] += 1
            else:
                two_counts[word] = 1
    if(target[i]=='3'):
        for word in words:
            if word in three_counts:
                three_counts[word] += 1
            else:
                three_counts[word] = 1
    if(target[i]=='4'):
        for word in words:
            if word in four_counts:
                four_counts[word] += 1
            else:
                four_counts[word] = 1
    if(target[i]=='5'):
        for word in words:
            if word in five_counts:
                five_counts[word] += 1
            else:
                five_counts[word] = 1
# Define a function to find posterior class probabilities :

Retrieving each row's sentence to test, to find the conditional probabilities of each word, word count of every word is appended into list. If the word found in test sentence isn't present in training dictionary, [2] laplace smoothing is performed by adding one to every existing feature. Finally, class posterior probability of every class is calculated by converting the word counts from list into probabilities and multiplying prior class probability. Class with the highest posterior probability is predicted class.

def find_cp(str,cdict,c):
    words = str.split(' ')
    calc=[]
    count=0
    den=math.ceil(train_size)
    for word in words:
        if word in cdict.keys() :
            #if(cdict[word]>5):
              #prob=prob*(cdict[word]/c)
              calc.append(cdict[word])
        else:
            #continue
            #prob=prob*(1/c)
            calc.append(1)
            count+=1
    den=den+count
    c=c+count
    prob=c/den
    for i in range(len(calc)):
        prob=prob*(calc[i]/c)

    return (prob)
# Step 4 :
# Fit the Model and Perform the Prediction :

Call the function to calculate the posterior probabilities of each class and the class with highest posterior probability is predicted. Accuracy is calculated based on the number of correct predictions.

acc=0
cp0=cp1=cp2=cp4=cp5=hpc=0
pc='3'
for i in range(math.ceil(train_size),math.ceil(dev_size)):
    str=datat[i]
    #words = str.split(' ')
    cp0=find_cp(str,zero_counts,c0)
    cp1=find_cp(str,one_counts,c1)
    cp2=find_cp(str,two_counts,c2)
    cp3=find_cp(str,three_counts,c3)
    cp4=find_cp(str,four_counts,c4)
    cp5=find_cp(str,five_counts,c5)
    hpc=max(cp0,cp1,cp2,cp3,cp4,cp5)
    if(hpc==cp0):
        pc='0'
    if(hpc==cp1):
        pc='1'
    if(hpc==cp2):
        pc='2'
    if(hpc==cp3):
        pc='3'
    if(hpc==cp4):
        pc='4'
    if(hpc==cp5):
        pc='5'
    #print(hpc)
    if(target[i]==pc):
        acc+=1

print('Accuracy upon predicting the Dev data set is ',acc/(math.ceil(dev_size)-math.ceil(train_size)))

acc2=0
cp0=cp1=cp2=cp4=cp5=hpc=0
for i in range(math.ceil(dev_size),math.ceil(test_size)):
    str=datat[i]
    #words = str.split(' ')
    cp0=find_cp(str,zero_counts,c0)
    cp1=find_cp(str,one_counts,c1)
    cp2=find_cp(str,two_counts,c2)
    cp3=find_cp(str,three_counts,c3)
    cp4=find_cp(str,four_counts,c4)
    cp5=find_cp(str,five_counts,c5)
    hpc=max(cp0,cp1,cp2,cp3,cp4,cp5)
    if(hpc==cp0):
        pc='0'
    if(hpc==cp1):
        pc='1'
    if(hpc==cp2):
        pc='2'
    if(hpc==cp3):
        pc='3'
    if(hpc==cp4):
        pc='4'
    if(hpc==cp5):
        pc='5'
    #print(hpc)
    if(target[i]==pc):
        acc2+=1

print('Accuracy upon predicting test data is ',acc2/(math.ceil(test_size)-math.ceil(dev_size)))


acc2=0
cp0=cp1=cp2=cp4=cp5=hpc=0
for i in range(math.ceil(dev_size),math.ceil(test_size)):
    str=datat[i]
    #words = str.split(' ')
    cp0=find_cp(str,zero_counts,c0)
    cp1=find_cp(str,one_counts,c1)
    cp2=find_cp(str,two_counts,c2)
    cp3=find_cp(str,three_counts,c3)
    cp4=find_cp(str,four_counts,c4)
    cp5=find_cp(str,five_counts,c5)
    hpc=max(cp0,cp1,cp2,cp3,cp4,cp5)
    if(hpc==cp0):
        pc='0'
    if(hpc==cp1):
        pc='1'
    if(hpc==cp2):
        pc='2'
    if(hpc==cp3):
        pc='3'
    if(hpc==cp4):
        pc='4'
    if(hpc==cp5):
        pc='5'
    #print(hpc)
    if(target[i]==pc):
        acc2+=1
print('Accuracy upon predicting test data is ',acc2/(math.ceil(test_size)-math.ceil(dev_size)))



