import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def prox(v,s):
    return max((1-(s/torch.norm(v)),0))*v

def convclust(X,W,v,gamma,eps):

    n=X.shape[1]
    p=X.shape[0]

    U1=torch.zeros(X.shape)
    U2=U1

    A=torch.zeros([n,p,n])
    V1=torch.zeros([n,p,n])

    V2=V1

    for i in range(0,n):
        for j in range(0,n):
            if W[i,j]>0:
                A[i][:,j]=1
                V1[i][:,j]=X[:,i]-X[:,j]

    for i in range(0,n):
        t=X[:,i]+sum(A[i].T)-sum(A)[:,i]+v*(sum(V1[i].T)-sum(V1)[:,i])
        U2[:,i]=(t+v*sum(X.T))/(1+n*v)

    for i in range(0,n):
        for j in range(0,n):
            if W[i,j]>0:
                V2[i][:,j]=prox(U2[:,i]-U2[:,j]-A[i][:,j]/v,(gamma*W[i,j]/v))
                A[i][:,j]=A[i][:,j]+v*(V2[i][:,j]-U2[:,i]+U2[:,j])

    U1=U2

    for i in range(0,n):
        t=X[:,i]+sum(A[i].T)-sum(A)[:,i]+v*(sum(V2[i].T)-sum(V2)[:,i])
        U2[:,i]=(t+v*sum(X.T))/(1+n*v)

    while torch.norm(U2-U1)>eps or torch.norm(V2-V1)>eps:
        
        V1=V2
        U1=U2

        for i in range(0,n):
            for j in range(0,n):
                if W[i,j]>0:
                    V2[i][:,j]=prox(U1[:,i]-U1[:,j]-A[i][:,j]/v,(gamma*W[i,j]/v))
                    A[i][:,j]=A[i][:,j]+v*(V2[i][:,j]-U1[:,i]+U1[:,j])

        for i in range(0,n):
            t=X[:,i]+sum(A[i].T)-sum(A)[:,i]+v*(sum(V2[i].T)-sum(V2)[:,i])
            U2[:,i]=(t+v*sum(X.T))/(1+n*v)

    return U2


def dist_matrix(X):
    X=torch.tensor(X)
    n=X.shape[1]
    D=torch.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            D[i,j]=torch.norm(X[:,i]-X[:,j])
    return D

def Ker_Conv_Clust(X,W,v=1,gamma=1,eps=0.0000001,h=1):

    n=X.shape[1]
    K=np.zeros([n,n])

    for i in range (0,n):
        for j in range (0,n):
            t=np.linalg.norm(X[:,i]-X[:,j])**2
            t=np.exp(-t/(2*(h**2)))
            K[i,j]=t

    U, diagonal, V_transpose = svd(K)

    D=np.diag(diagonal)
    D=np.sqrt(D)

    Z=torch.tensor(np.matmul(D,V_transpose))

    U=convclust(Z,W,v,gamma,eps)

    #return dist_matrix(U)

    return U

#KNN MATRIX

def smol(L,k):
    Index=[]
    Max=max(L)
    for i in range(0,k):
        a=L.index(min(L))
        Index.append(a)
        L[a]=Max+i+1
    return Index    
        

def KNN(X,k,h=1):
    n=X.shape[1]
    T=np.zeros([n,k])
    
    for i in range(0,n):
        dist=[]
        for j in range(0,n):
            dist.append(np.linalg.norm(X[:,i]-X[:,j])**2)
        T[i,]=smol(dist,k+1)[1:]
        
    W=np.zeros([n,n])
    
    for i in range(0,n):
        for j in range(0,n):
            if j in T[i,]:
                W[i,j]=(np.linalg.norm(X[:,i]-X[:,j])**2)/(2*h)
                W[i,j]=math.exp(-W[i,j])

    return W

#Kernel Convex Cluster with KNN matrix

def KerConvKNN(X,v=1,k=6,gamma=1,eps=0.0000001,h1=1,h2=1):
    W=KNN(X,k,h2)

    W=torch.tensor(W)

    W=(W+W.T)/2

    U=Ker_Conv_Clust(X,W,v,gamma,eps,h1)

    return U

#GRAPH CLUSTERING

def grph(X,delta):
    n=X.shape[0]

    D=np.zeros((n,n))

    for i in range(0,n):
        for j in range(0,n):
            if X[i,j]<delta:
                D[i,j]=X[i,j]

    return D

#To get the optimal number of clusters




def WCSS(X):        

    #n=X.shape[1]
    mean = np.mean(X, axis=1)
    X1 = X - mean[:, None]
    #mean=sum(X.T)/n

    X1=(X.T-mean).T

    Var=np.linalg.norm(X1)**2

    return Var

def Elbow1(X,k_max=15,v=1,h1=1,h2=1,gamma=1):
    U=KerConvKNN(X,v=v,h1=h1,h2=h2,gamma=gamma)

    U=np.array(U)

    SSE_list=[0]*(k_max)

    for n in range(1,k_max+1):
        clustering = AgglomerativeClustering(n_clusters=n).fit(U.T)
        l=clustering.labels_

        for i in range(0,n):
            a=np.array(np.where(l==i))       #indices of all the points with cluster identity i
            p=a.shape[1]
            b=[]
            for x in range(0,p):
                b.append(a[0][x])
            t=U[:,b]
            SSE_list[n-1]+=WCSS(t)
            #SSE_list[n-1]+=np.linalg.norm(np.array(dist_matrix(t)))/(2**p)

    return SSE_list       
    

def Elbowplot(X,k=15,v=1,h1=1,h2=1,gamma=1):
    t=Elbow1(X,k,v,h1,h2,gamma)
    n=[i for i in range(1,k+1)]
    plt.plot(n,t,color='red')
    plt.show()
            
    
#import pandas as pd

#df = pd.read_csv("C:\\Users\\S Pan\\Downloads\\appendicitis\\appendicitis.csv", delimiter='\t')




#df=np.asarray(df)

#df

#df=df[9:,]    Take the corresponding rows

#df.shape

#DF=np.zeros((85,9))
#for i in range(0,85):
#    a=df[i]
#    string_list = a[0].split(',')
#    float_list = [float(x) for x in string_list]
#    DF[i]=float_list

#X=DF[:,0:16]

#Y=DF[:,16]    #Take the corresponding response column

#RESULTS



import os
path="C:\\Users\\S Pan\\Downloads"
os.chdir(path)

import scipy.io
mat = scipy.io.loadmat('lymphoma.mat')
#mat
#mat.keys()
X=mat['X']
import scipy.stats as ss
#X=np.array(ss.zscore(X))
#X=X.T
#U=KerConvKNN(X)

X=X.T
U=KerConvKNN(X)


Y=mat['Y']
l=len(Y)
y=[]
for i in range(0,l):
    y.append(Y[i][0])

Y=y
#U=KerConvKNN(X,v=0.005,h1=10,h2=10,gamma=0.1)



#POST PROCESSING STEPS

#D=dist_matrix(U)
#plt.imshow(D)
#plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

#U=U.T
#dendrogram(linkage(U, 'ward'))

#plt.show()

from graph_based_clustering import ConnectedComponentsClustering

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from kkmeans import kkmeans



#clustering = AgglomerativeClustering(n_clusters=3).fit(U)

#print(clustering.labels_)

import sklearn.metrics

from sklearn.metrics.cluster import normalized_mutual_info_score

#sklearn.metrics.normalized_mutual_info_score(Y,clustering.labels_, average_method='arithmetic')

def NMI(X,Y,n,Ker=1,v=1,h1=1,h2=1,gamma=1):
    if Ker==1:
        U=KerConvKNN(X,v=v,h1=h1,h2=h2,gamma=gamma)
    else:
        U=convclust(X,KNN(X,6,h2),v,gamma,0.0000001)
    clustering = AgglomerativeClustering(n_clusters=n).fit(U.T)
    t=sklearn.metrics.normalized_mutual_info_score(Y,clustering.labels_, average_method='arithmetic')
    return t

def NMI_1(X,Y,n,Ker=1,v=1,h1=1,h2=1,gamma=1):
    t=NMI(X,Y,n,Ker=1,v=v,h1=h1,h2=h2,gamma=gamma)
    print("Kernel Convex Clustering:",t)
    print()

    t=NMI(X,Y,n,Ker=0,v=v,h1=h1,h2=h2,gamma=gamma)
    print("Convex Clustering:",t)
    print()

    #kkmeans = kkmeans(n_clusters=2, kernel=rbf_kernel(gamma=0.1))
    #labels=kkmeans.predict(X.T)
    #t=sklearn.metrics.normalized_mutual_info_score(Y,labels, average_method='arithmetic')
    #print("kernel K means Clustering:",t)
    #print()

    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit_predict(X.T)
    t=sklearn.metrics.normalized_mutual_info_score(Y,kmeans, average_method='arithmetic')
    print("k means Clustering:",t)
    print()

    clustering = AgglomerativeClustering(n_clusters=n).fit(X.T)
    t=sklearn.metrics.normalized_mutual_info_score(Y,clustering.labels_, average_method='arithmetic')
    print("Hierarchical Clustering:",t)
    print()
    
    clustering = SpectralClustering(n_clusters=n,assign_labels='discretize',random_state=0).fit(X.T)
    t=sklearn.metrics.normalized_mutual_info_score(Y,clustering.labels_, average_method='arithmetic')
    print("Spectral Clustering:",t)

    
    
    
