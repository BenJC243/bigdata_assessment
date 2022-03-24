## K-MEANS CLUSTERING
#load relevant modules 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#import cleaned data 
myfile1='cleaned_data_FINAL.csv'
df=pd.read_csv(myfile1)

#create object w/ numerical column list 
col_heads=list(df.columns)
catlist= df.select_dtypes(object).columns.values.tolist()
numlist= df.select_dtypes(np.number).columns.values.tolist()

for col in numlist:
    if col != 'encounter_id'and col[-2:] =='id':
        numlist.remove(col)
        catlist.append(col)
numlist.remove('patient_nbr')

#min/max normalisations and dropping non-continuous variables from data frame 
df_norm=df.copy()
for x in numlist: 
    if x in df.columns:
        df_norm[x]= (df[x]-df[x].min()) / (df[x].max()-df[x].min())
df_norm.loc[1]

#remove categorical values from df, and also patient numbers/ids 
for x in catlist:
    if x in df_norm.columns:
        df_norm.drop(labels=x,axis=1, inplace=True)
df_norm.drop(labels=['patient_nbr','admission_source_id'],axis=1,inplace=True)

### model building 
#call algorithm with 6 clusters 
model = KMeans(n_clusters=6)
model.fit(df_norm)
print('cost=',model.inertia_) # J score (lower = better)
#j score = sum of square distances between each point + its centroid 
print(model.labels_) #labels denoting clusters for each row of df 

#add cluster assignments to df with below script 
labels=model.labels_ #take titles of each labelled datapoint 
md=pd.Series(labels) #make a series out of each datapoint 
df['clust']=md
df_norm['clust']=md
# means of data points for each cluster 
df_norm.groupby('clust').mean()
# see the mean of each column for each cluster 

#make elbow plot - for determining / supporting K value
#identify 'joint' in elbow that informs on optimal cluster no. 
def elbow(data):
    print("\nPlotting elbow method...")
    sse = {}
    for k in range(2, 20, 2):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        print(k, kmeans.inertia_)
        sse[k] = kmeans.inertia_  
        # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()), linewidth=4)
    plt.xlabel("Number of clusters")
    plt.ylabel("Cost")
    plt.show()
    print("DONE")
elbow(df_norm) # use the above function to print an elbow plot for our data 

##PCA 
######## 2D plot of the clusters
##create PCA model 
pca_data = PCA(n_components=2).fit(df_norm)
pca_2d = pca_data.transform(df_norm)

#create new column for readmission data to overlay on scatterplot 
df.readmitted=pd.Categorical(df.readmitted) #create category of readmittance to colour the plot
df['code'] = df.readmitted.cat.codes #0 = <30, 1 = >30, 2= NO

#plot scatterplot of clusters using PCAs 
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=labels) #change c= to labels/df['code'] for different colours of clusters 
#plt.legend([0,1,2],['Before 30d', 'After 30d','Not readmitted']) #0 = <30, 1 = >30, 2= NO
plt.title('Patient clusters / all continuous variables / k = 6')
plt.show()

# second graph w/ readmitted status overlayed
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=df['code']) #change c= to labels/df['code'] for different colours of clusters 
#plt.legend([0,1,2],['Before 30d', 'After 30d','Not readmitted']) #0 = <30, 1 = >30, 2= NO
plt.title('Patient clusters / all continuous variables / k = 6')
plt.show()

##IMPROVEMENTS TO THE K-MEANS MODEL (using PCA)
df_norm.drop(labels=['encounter_id'],axis=1,inplace=True) #DROP THIS - explains vast majority of variance 
pca=PCA()
pca.fit(df_norm)

pca.explained_variance_ratio_ #show the amount of variance each column explains

#cumulative variance plot to determine no. of features to include in PCA 
##CUMULATIVE VARIANCE PLOT 
plt.plot(range(1,10),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')
plt.title('explained variance by components')
plt.xlabel('no. of components')
plt.ylabel('cumulative explained variance')

pca2 = PCA(n_components=5) #build new PCA model 
pca2.fit(df_norm) #fit our data to PCA model 
scores_pca2=pca2.transform(df_norm) #list scores from model 

#calling values from new model 
newmodel=KMeans(n_clusters=8,init='k-means++',random_state=50) #create new KMeans model for PCA data 
newmodel.fit(scores_pca2)

j=newmodel.inertia_ #get j-score for new model 

##script end 


