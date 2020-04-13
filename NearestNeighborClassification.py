#Please put your code for Step 2 and Step 3 in this file.
#Cynthia Jelke
#I worked alone on this
#This took me 2 1/2 hours

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy import stats


# FUNCTIONS

#reads files as arrays for glucose, hemoglobin, and their classification
def openckdfile():
    glucose, hemoglobin, classification = np.loadtxt('ckd.csv', delimiter=',', skiprows=1, unpack=True)
    return glucose, hemoglobin, classification

#put convert arrays so that they're all in the same scale (between 0 and 1)
def normalizeData(glucose, hemoglobin, classification):
    h_scaled = (hemoglobin - 3.1)/(17.8 - 3.1)
    g_scaled = (glucose - 70)/(490 - 70)
    return g_scaled, h_scaled, classification

#graph the data with its classifications
def graphData(glucose, hemoglobin, classification):
    plt.figure()
    plt.plot(hemoglobin[classification==1],glucose[classification==1], "k.", label = "Class 1")
    plt.plot(hemoglobin[classification==0],glucose[classification==0], "r.", label = "Class 0")
    plt.xlabel("Hemoglobin")
    plt.ylabel("Glucose")
    plt.legend()
    plt.show()
    
#generate a new random point
def createTestCase():
    newglucose = random.random()
    newhemoglobin = random.random()
    return newglucose, newhemoglobin

#finds the distance between each data point and the random point
def calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin):
    
    #initialize array
    distance = np.zeros(len(glucose), dtype=float)
    
    #distance formula
    for i in range(len(glucose)):
        distance[i] = math.sqrt((newhemoglobin - hemoglobin[i])**2 + (newglucose - glucose[i])**2)
    return distance

#classifies the random point based on its closest neighbor
def nearestNeighborClassifier(newglucose, newhemoglobin, glucose, hemoglobin, classification):
    distance = calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
    #finds the closest neighbor
    min_index = np.argmin(distance)
    nearest_class = classification[min_index]
    return nearest_class

#classifies the random point based on a chosen amount of neighbors
def k_nearestClassification(newglucose, newhemoglobin, glucose, hemoglobin, classification, k):
    distance = calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)
    sorted_indices = np.argsort(distance)
    #breaks it into an array of its k-nearest neighbors
    k_indices = sorted_indices[:k]
    k_classification = classification[k_indices]
    #picks 1 or 0 depending on which appears more often
    nearest_class = stats.mode(k_classification)
    return nearest_class

#graphs the location of the test case and the dataset
def graphTestCase(newglucose, newhemoglobin, glucose, hemoglobin, classification, nearest_class):
     plt.figure()
     #0 is non-CKD and 1 is CKD
     plt.plot(hemoglobin[classification==1],glucose[classification==1], "k.", label = "Class 1")
     plt.plot(hemoglobin[classification==0],glucose[classification==0], "r.", label = "Class 0")
     #plots the random point in the shape of a diamond
     if(nearest_class[0] == 1.):
         plt.plot(newhemoglobin, newglucose, "kd")
     else:
         plt.plot(newhemoglobin, newglucose, "rd")
     plt.xlabel("Hemoglobin")
     plt.ylabel("Glucose")
     plt.legend()
     plt.show()
    

# MAIN SCRIPT
     
#how many neighbors you're comparing the point to
k = 3

#read in data
glucose, hemoglobin, classification = openckdfile()
#put data in scale
glucose, hemoglobin, classification = normalizeData(glucose, hemoglobin, classification)

#creates the new data point
newglucose, newhemoglobin = createTestCase()

#find the distances
distance = calculateDistanceArray(newglucose, newhemoglobin, glucose, hemoglobin)

#classifies the random data point
nearest_class = k_nearestClassification(newglucose, newhemoglobin, glucose, hemoglobin, classification, k)

#classifies the random data point based on one neighbor (uncomment line below)
#nearest_class = nearestNeighborClassifier(newglucose, newhemoglobin, glucose, hemoglobin, classification)

#graphs the test case amongst the data points
graphTestCase(newglucose, newhemoglobin, glucose, hemoglobin, classification, nearest_class)
