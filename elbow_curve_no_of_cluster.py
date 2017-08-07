###############
# Reference
# http://www.awesomestats.in/python-cluster-validation/
###############

import pandas as pd

#fetching R datasets to use in python
import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
from rpy2.robjects import pandas2ri
pandas2ri.activate()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

%matplotlib inline

R = ro.r

df = conversion.ri2py(R['mtcars'])
print(df.head())

from sklearn.cluster import KMeans

df.columns

#get data 
X = df[['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']]

#import standard Scaler to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform( X )

#set the number of cluser range assumption
cluster_range = range( 1, 20 )
cluster_errors = []

#run the algo to get the clustering done
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters )
  clusters.fit( X_scaled )
  cluster_errors.append( clusters.inertia_ )

#create a data frame to check the number of cluster and corresponding error
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:10]

#plot the eblow curve to visuvalize the results
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )

#take the last point as number of cluster to be used in clustring, where you can it is a drastic change from the previous point
