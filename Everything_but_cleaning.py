#load in relevant modules
import pandas as pd 
import numpy as np
import pandas as pd 
from scipy import stats

##0 - importing the data 
myfile1='/home/fic2/Desktop/Big_Data/diabetic_data/cleaned_data_FINAL.csv'
df=pd.read_csv(myfile1)

df['readmitted'] = df['readmitted'].replace({'>30':'1', '<30':'1', 'NO': '0'})
dum_cols=['gender', 'race']
dummies=pd.get_dummies(df, columns=dum_cols)
df = pd.concat([df, dummies], axis = 1)
df = df.loc[:,~df.columns.duplicated()]
df.shape

ls = list(df['diag_1'])
def diag_1(df):
    for i, x in enumerate(df['diag_1']):
        v = str(x)
        if v[0].isalpha():
            ls[i] = v.replace(v,'other')
        elif v[0].isdigit():
            s = float(v)
            if (s > 390 and s < 460) or s == 785:#s in range(390, 460):
                p = str(s)
                ls[i] = p.replace(p,'circulatory_disease')
            elif s > 249.99 and s < 251:
                p = str(s)
                ls[i] = p.replace(p,'diabetes_mellitus')
            elif (s > 459 and s < 520) or s == 786:
                p = str(s)
                ls[i] = p.replace(p,'respiratory_disease')
            elif (s > 519 and s < 580) or s == 787:
                p = str(s)
                ls[i] = p.replace(p,'gastrointestinal_disease')
            elif (s > 799 and s < 1000):
                p = str(s)
                ls[i] = p.replace(p,'injury/poisoning')
            elif (s > 709 and s < 740):
                p = str(s)
                ls[i] = p.replace(p,'musculoskeletal/connective')
            elif (s > 579 and s < 740):
                p = str(s)
                ls[i] = p.replace(p,'genitourinary')
            elif (s > 139 and s < 240):
                p = str(s)
                ls[i] = p.replace(p,'neoplasms')
            else:
                p = str(s)
                ls[i] = p.replace(p,'other')                

    df['diag_1'] = ls
    return df['diag_1']

df['diag_1'] = diag_1(df)

ls1 = list(df['diag_2'])
def diag_2(df):
    for i, x in enumerate(df['diag_2']):
        v = str(x)
        if v[0].isalpha():
            ls1[i] = v.replace(v,'other')
        elif v[0].isdigit():
            s = float(v)
            if (s > 390 and s < 460) or s == 785:#s in range(390, 460):
                p = str(s)
                ls1[i] = p.replace(p,'circulatory_disease')
            elif s > 249.99 and s < 251:
                p = str(s)
                ls[i] = p.replace(p,'diabetes_mellitus')
            elif (s > 459 and s < 520) or s == 786:
                p = str(s)
                ls1[i] = p.replace(p,'respiratory_disease')
            elif (s > 519 and s < 580) or s == 787:
                p = str(s)
                ls1[i] = p.replace(p,'gastrointestinal_disease')
            elif (s > 799 and s < 1000):
                p = str(s)
                ls1[i] = p.replace(p,'injury/poisoning')
            elif (s > 709 and s < 740):
                p = str(s)
                ls1[i] = p.replace(p,'musculoskeletal/connective')
            elif (s > 579 and s < 740):
                p = str(s)
                ls1[i] = p.replace(p,'genitourinary')
            elif (s > 139 and s < 240):
                p = str(s)
                ls1[i] = p.replace(p,'neoplasms')
            else:
                p = str(s)
                ls1[i] = p.replace(p,'other')                

    df['diag_2'] = ls
    return df['diag_2']

df['diag_2'] = diag_2(df)

ls2 = list(df['diag_3'])
def diag_3(df):
    for i, x in enumerate(df['diag_3']):
        v = str(x)
        if v[0].isalpha():
            ls1[i] = v.replace(v,'other')
        elif v[0].isdigit():
            s = float(v)
            if (s > 390 and s < 460) or s == 785:#s in range(390, 460):
                p = str(s)
                ls2[i] = p.replace(p,'circulatory_disease')
            elif s > 249.99 and s < 251:
                p = str(s)
                ls2[i] = p.replace(p,'diabetes_mellitus')
            elif (s > 459 and s < 520) or s == 786:
                p = str(s)
                ls2[i] = p.replace(p,'respiratory_disease')
            elif (s > 519 and s < 580) or s == 787:
                p = str(s)
                ls2[i] = p.replace(p,'gastrointestinal_disease')
            elif (s > 799 and s < 1000):
                p = str(s)
                ls2[i] = p.replace(p,'injury/poisoning')
            elif (s > 709 and s < 740):
                p = str(s)
                ls2[i] = p.replace(p,'musculoskeletal/connective')
            elif (s > 579 and s < 740):
                p = str(s)
                ls2[i] = p.replace(p,'genitourinary')
            elif (s > 139 and s < 240):
                p = str(s)
                ls2[i] = p.replace(p,'neoplasms')
            else:
                p = str(s)
                ls2[i] = p.replace(p,'other')                

    df['diag_3'] = ls
    return df['diag_3']
df['diag_3'] = diag_3(df)

import matplotlib.pyplot as plt
import seaborn as sns

def readmission_hists(df, plot_cols, grid_col):
    for col in plot_cols:
        #if (col != 'encounter_id' and col != 'patient_nbr'):
        if col == 'age':
            g = sns.FacetGrid(df, col=grid_col, margin_titles=True)
            g.map(plt.hist, col)
            plt.show()
readmission_hists (df, df.select_dtypes(include=[np.number]).columns, "readmitted")

ls3=[]
for i, j in zip(list(df['readmitted']), list(df['gender_Male'])):
    if int(i) == int(j):
        ls3.append(i)
#print(len(ls3))
print(ls3.count('1'), 'male') # = num of men readmitted
#print(ls3.count('0'))

ls4=[]
for i, j in zip(list(df['readmitted']), list(df['gender_Female'])):
    if int(i) == int(j):
        ls4.append(i)
print(ls4.count('1'), 'female')

ls5=[]
for i, j in zip(list(df['readmitted']), list(df['race_Caucasian'])):
    if int(i) == int(j):
        ls5.append(i)
print(ls5.count('1'), 'cauc')

ls6=[]
for i, j in zip(list(df['readmitted']), list(df['race_AfricanAmerican'])):
    if int(i) == int(j):
        ls6.append(i)
print(ls6.count('1'), 'african-americ')

ls7=[]
for i, j in zip(list(df['readmitted']), list(df['race_Asian'])):
    if int(i) == int(j):
        ls7.append(i)
print(ls7.count('1'), 'asian')

ls8=[]
for i, j in zip(list(df['readmitted']), list(df['race_Hispanic'])):
    if int(i) == int(j):
        ls8.append(i)
print(ls8.count('1'), 'hispanic')

print(df['gender_Male'].value_counts())
print(df['gender_Female'].value_counts())
print(df['race_Caucasian'].value_counts())
print(df['race_AfricanAmerican'].value_counts())
print(df['race_Asian'].value_counts())
print(df['race_Hispanic'].value_counts())

#normalised --> (total of group that were readmitted)/(total of group)
x = ['Male', 'Female']
y = [11410 / 30191, 13450 / 34274]
sns.barplot(x, y)
plt.show()

x = ['Caucasian', 'African-American','Asian','Hispanic']
y = [18997/48048, 4397/11718, 140/461, 449/1341]

sns.barplot(x, y)
plt.show()

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'circulatory_disease':
        ls9.append(1)
print(ls9.count(1), 'circulatory')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'respiratory_disease':
        ls9.append(1)
print(ls9.count(1), 'respiratory')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'gastrointestinal_disease':
        ls9.append(1)
print(ls9.count(1), 'gastrointestinal_disease')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'genitourinary':
        ls9.append(1)
print(ls9.count(1), 'genitourinary')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'diabetes_mellitus':
        ls9.append(1)
print(ls9.count(1), 'diabetes_mellitus')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'musculoskeletal/connective':
        ls9.append(1)
print(ls9.count(1), 'musculoskeletal/connective')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'injury/poisoning':
        ls9.append(1)
print(ls9.count(1), 'injury/poisoning')
        
ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'neoplasms':
        ls9.append(1)
print(ls9.count(1), 'neoplasms')

ls9=[]
for i, j in zip(list(df['readmitted']), list(df['diag_1'])):
    if i == '1' and j == 'other':
        ls9.append(1)
print(ls9.count(1), 'other')


#df['readmitted']

df['diag_1'].value_counts()

import seaborn as sns
#from matplotlib import pyplot
x = ['Circulatory', 'Respiratory', 'Gastrointestinal','Genitourinary','Diabetes',
    'musculoskeletal/connective', 'injury/poisoning', 'Neoplasms', 'other']
y = [8025/19652, 3581/9006, 2274/6077, 1926/5345, 1959/4690, 1319/3847, 1603/4361, 755/2544, 3418/8946]

sns.set(rc={'figure.figsize':(20,7.5)})
sns.barplot(x, y)#, height=5)
plt.show()

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'circulatory_disease':
        ls11.append(1)
print(ls11.count(1), 'circulatory')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'respiratory_disease':
        ls11.append(1)
print(ls11.count(1), 'respiratory')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'gastrointestinal_disease':
        ls11.append(1)
print(ls11.count(1), 'gastrointestinal_disease')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'genitourinary':
        ls11.append(1)
print(ls11.count(1), 'genitourinary')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'diabetes_mellitus':
        ls11.append(1)
print(ls11.count(1), 'diabetes_mellitus')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'musculoskeletal/connective':
        ls11.append(1)
print(ls11.count(1), 'musculoskeletal/connective')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'injury/poisoning':
        ls11.append(1)
print(ls11.count(1), 'injury/poisoning')
        
ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'neoplasms':
        ls11.append(1)
print(ls11.count(1), 'neoplasms')

ls11=[]
for i, j in zip(list(df['readmitted']), list(df['diag_2'])):
    if i == '1' and j == 'other':
        ls11.append(1)
print(ls11.count(1), 'other')


#df['readmitted']

print(df['diag_2'].value_counts())

#DIAG 2 & 3 (same num of diagnoses)
x = ['Circulatory disease', 'Respiratory disease', 'Gastrointestinal disease','Genitourinary disease','Diabetes',
    'musculoskeletal/connective', 'injury/poisoning', 'neoplasms', 'other']
y = [7012/17205, 3047/7504, 1937/5137, 1561/4254, 5247/13691, 1031/2941, 0, 679/2241, 2928/7625]

sns.set(rc={'figure.figsize':(20,7.5)})
sns.barplot(x, y)
plt.show()

#Model 1

from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import feature_selection#, metrics
from sklearn.feature_selection import RFE 
from sklearn.svm import SVR
#df = df.loc[:,~df.columns.duplicated()] #remove duplicated 'readmitted' col
df1 = df[['encounter_id','num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 
'number_inpatient', 'age', 'num_lab_procedures', 'number_diagnoses', 
'num_procedures','diag_1', 'diag_2', 'diag_3', 'readmitted']]
df1 = df1.drop(df1.loc[df1['diag_1']!='diabetes_mellitus'].index)#, inplace=False)
df1 = df1.drop(df1.loc[df1['diag_2']!='diabetes_mellitus'].index)#, inplace=False)
df1 = df1.drop(df1.loc[df1['diag_3']!='diabetes_mellitus'].index)#, inplace=False)

model1 = linear_model.LogisticRegression()
cols = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 
'number_inpatient', 'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'encounter_id']

X = df1[cols]
Y = df1['readmitted']
model1.fit(X, Y)

print('Model score:\n ', model1.score(X,Y))
print('Coefficients: ')
for feat, coef in zip(cols, model1.coef_[0]):
    print(feat, coef)

X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.25)
model2 = linear_model.LogisticRegression()
model2.fit(X_train, Y_train)
print("Score against training data: ",model2.score(X_train, Y_train))
print("Score against test data: ", model2.score(X_test, Y_test))
#scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy', cv=10)
#print("Cross validation mean scores: {}".format(scores.mean()))

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

df1['readmitted'] = df1['readmitted'].replace({1:'1', 1:'1', 0: '0'})
#df1['readmitted'] = df1['readmitted'].replace({'1':1, '0':0})

#print(df1)
#print(df['readmitted'])#.dtype)
pred_test = model2.predict(X_test)
pred_train = model2.predict(X_train)

## Acuracy score for the training data
accuracy_train = accuracy_score(pred_train, Y_train)
print('Accuracy for the training set: ', accuracy_train)
## Acuracy score for the test data
accuracy_test = accuracy_score(pred_test, Y_test)
print('Accuracy for the test set: ', accuracy_test)


# confusion matrix for the test data
pred = pred_test
cm = confusion_matrix(Y_test, pred)

TN, FP, FN, TP = confusion_matrix(Y_test, pred_test).ravel()

print('True Positives: ', TP)
print('False Positives: ', FP)
print('True Negatives: ', TN)
print('False Negatives: ', FN)

accuracy = (TP+TN) /(TP+FP+TN+FN)

print('Accuracy score = ', accuracy)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model2.classes_)
disp.plot()
plt.show()

# Calculate Accuracy, Precision and Recall Metrics for the test data
accuracy = accuracy_score(pred, Y_test)
print('Accuracy: ', accuracy)
precision = precision_score(pred, Y_test,pos_label = '1')
print('Precision: ', precision)
recall = recall_score(pred, Y_test,pos_label = '1')
print('Recall: ', recall)
f1score = f1_score(pred, Y_test,pos_label = '1')

print('F1_score: ', f1score)
plt.show()

   #CROSS VALIDATION
scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy')
print('Cross validation mean score: ', scores.mean())

#Improved model
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import feature_selection#, metrics
from sklearn.feature_selection import RFE 
from sklearn.svm import SVR
#df = df.loc[:,~df.columns.duplicated()] #remove duplicated 'readmitted' col
df1 = df[['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 
'number_inpatient', 'age', 'num_lab_procedures', 'number_diagnoses', 
'num_procedures','diag_1', 'diag_2', 'diag_3', 'readmitted']]
df1 = df1.drop(df1.loc[df1['diag_1']!='diabetes_mellitus'].index)#, inplace=False)
df1 = df1.drop(df1.loc[df1['diag_2']!='diabetes_mellitus'].index)#, inplace=False)
df1 = df1.drop(df1.loc[df1['diag_3']!='diabetes_mellitus'].index)#, inplace=False)

model1 = linear_model.LogisticRegression()
cols = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 
'number_inpatient', 'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures']

X = df1[cols]
Y = df1['readmitted']
model1.fit(X, Y)

print('Model score:\n ', model1.score(X,Y))
print('Coefficients: ')
for feat, coef in zip(cols, model1.coef_[0]):
    print(feat, coef)

X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.25)
model2 = linear_model.LogisticRegression()
model2.fit(X_train, Y_train)
print("Score against training data: ",model2.score(X_train, Y_train))
print("Score against test data: ", model2.score(X_test, Y_test))
#scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy', cv=10)
#print("Cross validation mean scores: {}".format(scores.mean()))

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

df1['readmitted'] = df1['readmitted'].replace({1:'1', 1:'1', 0: '0'})
#df1['readmitted'] = df1['readmitted'].replace({'1':1, '0':0})

#print(df1)
#print(df['readmitted'])#.dtype)
pred_test = model2.predict(X_test)
pred_train = model2.predict(X_train)

## Acuracy score for the training data
accuracy_train = accuracy_score(pred_train, Y_train)
print('Accuracy for the training set: ', accuracy_train)
## Acuracy score for the test data
accuracy_test = accuracy_score(pred_test, Y_test)
print('Accuracy for the test set: ', accuracy_test)


# confusion matrix for the test data
pred = pred_test
cm = confusion_matrix(Y_test, pred)

TN, FP, FN, TP = confusion_matrix(Y_test, pred_test).ravel()

print('True Positives: ', TP)
print('False Positives: ', FP)
print('True Negatives: ', TN)
print('False Negatives: ', FN)

accuracy = (TP+TN) /(TP+FP+TN+FN)

print('Accuracy score = ', accuracy)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model2.classes_)
disp.plot()
plt.show()

# Calculate Accuracy, Precision and Recall Metrics for the test data
accuracy = accuracy_score(pred, Y_test)
print('Accuracy: ', accuracy)
precision = precision_score(pred, Y_test,pos_label = '1')
print('Precision: ', precision)
recall = recall_score(pred, Y_test,pos_label = '1')
print('Recall: ', recall)
f1score = f1_score(pred, Y_test,pos_label = '1')

print('F1_score: ', f1score)
plt.show()

   #CROSS VALIDATION
scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring='accuracy')
print('Cross validation mean score: ', scores.mean())
