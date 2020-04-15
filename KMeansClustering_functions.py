#Please place your FUNCTION code for step 4 here.
#Cynthia Jelke
#The functions openckdfile, select, assign, and graphingKMeans come from
#Jenn's helper code
#This took me 6 1/2 hours

import numpy as np
import matplotlib.pyplot as plt
import random

#reads files as arrays for glucose, hemoglobin, and their classification
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    hemoglobin = (hemoglobin - 3.1)/(17.8 - 3.1)
    glucose = (glucose - 70)/(490 - 70)
    return glucose, hemoglobin, classification

#chooses k number of random centroids
def select(K):
    return np.random.random((K, 2))


#sets the data points as part of clusters based on distance from centroids
def assign(centroids, hemoglobin, glucose):

    K = centroids.shape[0]
    
    #initializes an array to hold the distances
    distances = np.zeros((K, len(hemoglobin)))

    #calculates the distances between each centroid and all the data points
    for i in range(K):

        g = centroids[i,1]
        h = centroids[i,0]

        distances[i] = np.sqrt((hemoglobin-h)**2+(glucose-g)**2)
 
    #assigns the data points to clusters based on which distance is shortest
    assignments = np.argmin(distances, axis = 0)    

    #print(assignments)
    return assignments


#finds the mean point (center) of a cluster
def findClusterMean(assignments, hemoglobin, glucose, clusternum):
    #counts how many of the datapoints are in a certain cluster
    quantity = np.count_nonzero(assignments == clusternum)
    #finds and creates an array of the indices of each of those data points
    locations = np.where(assignments == clusternum)[0]
    #takes the length of that array
    length = len(locations)
    
    #initializes arrays to store the location of each data point within
    #the cluster
    c_hemoglobin = np.zeros(length, dtype = float)
    c_glucose = np.zeros(length, dtype = float)
    
    #fills the arrays initialized above with the location values
    for i in range(length):
        c_hemoglobin[i] = hemoglobin[locations[i]]
        c_glucose[i] = glucose[locations[i]]
        
    #calculates the center location of each cluster
    hem_mean = np.sum(c_hemoglobin)/quantity
    glu_mean = np.sum(c_glucose)/quantity
    
    #assigns the center location to a new centroid
    new_centroid = np.array([hem_mean,glu_mean])
    return new_centroid


#applies above functions to properly use K Means Clustering to assign 
#clusters
def kMeansClustering(k, iteration_count):
    glucose, hemoglobin, classification = openckdfile()

    #assigns a random set of K centroids
    centroids = select(k)
    
    j = 0
    #uses blind iteration so runs a set number of times
    while j <= iteration_count:
        #assigns the data points to clusters based on centroid location
        assignments = assign(centroids, hemoglobin, glucose)
    
        #calculates the center value of the cluster and replaces the 
        #centroid array values with those values
        for i in range(k):
            new_centroid = findClusterMean(assignments, hemoglobin, glucose, i)
            centroids[i] = new_centroid
        #increments j
        j += 1
    print("The final centroids are at:")
    print(centroids)
    return assignments

#graphs the data points in their assign clusters
def graphingKMeans(glucose, hemoglobin, assignment, centroids, k):
    plt.figure()
    for i in range(assignment.max()+1):
        rcolor = np.random.rand(4,)
        plt.plot(hemoglobin[assignment==i],glucose[assignment==i], ".", label = "Class " + str(i), color = rcolor)
        #display the original centroid location
        if(k <= 5):
            plt.plot(centroids[i, 0], centroids[i, 1], "D", label = "Centroid " + str(i), color = rcolor)
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.show()
    

#Calculates the True Negative Rate and the False Negative Rate
def NegativesRate(classification, assignments, hemoglobin):
    #find the point with the highest hemoglobin
    neg_loc = np.argmax(hemoglobin)
    #finds whether that point has been assigned as group 1 or 0 
    neg_val = assignments[neg_loc]
    
    #counts how many patients actually are non-CKD
    tot_num = np.count_nonzero(classification == 0)
    
    true_count = 0
    false_count = 0
    
    #if the assignment matches the actal non-CKD classification add
    #one to the true counter, else if the algorithm assigned a point as
    #non-CKD and they actually have CKD increment false counter by 1
    for i in range(len(hemoglobin)):
        if(classification[i] == 0 and assignments[i] == neg_val):
            true_count += 1
        elif(classification[i] == 1 and assignments[i] == neg_val):
            false_count += 1
            
    #calculates the true and false negative rates
    true_rate = round(((true_count/tot_num)*100),3)
    false_rate = round(((false_count/(len(hemoglobin) - tot_num))*100),3)
    return true_rate, false_rate


#Calculates the True Posative Rate and the False Posative Rate
def PosativesRate(classification, assignments, hemoglobin):
    #find the point with the lowest hemoglobin
    pos_loc = np.argmin(hemoglobin)
    #finds whether that point has been assigned as group 1 or 0 
    pos_val = assignments[pos_loc]
    
    #counts how many patients actually are CKD
    tot_num = np.count_nonzero(classification == 1)
    
    true_count = 0
    false_count = 0
    
    #if the assignment matches the actal CKD classification add
    #one to the true counter, else if the algorithm assigned a point as
    #CKD and they actually are non-CKD increment false counter by 1
    for i in range(len(hemoglobin)):
        if(classification[i] == 1 and assignments[i] == pos_val):
            true_count += 1
        elif(classification[i] == 0 and assignments[i] == pos_val):
            false_count += 1
            
    #calculates the true and false positive rates
    true_rate = round(((true_count/tot_num)*100),3)
    false_rate = round(((false_count/(len(hemoglobin) - tot_num))*100),3)
    return true_rate, false_rate

########################################## MAIN ####################


