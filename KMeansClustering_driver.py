#Please place your FUNCTION code for step 4 here.
#Cynthia Jelke
#This took me 5 minutes

import KMeansClustering_functions as kmc #Use kmc to call your functions
    

#reads in arrays for glucose, hemoglobin, and classification
glucose, hemoglobin, classification = kmc.openckdfile()


#change this value for how many clusters you want
k = 3

#assign k random starting centroids
centroids = kmc.select(k)

#runs the algorithm to assign data points to clusters
assignments = kmc.kMeansClustering(k, 10000)

#graphs the clusters
kmc.graphingKMeans(glucose, hemoglobin, assignments, centroids, k)

#prints the original centroid locations
print("The centroids are originally at:")
print(centroids)

#prins the true and false Negatives/Posatives rate
print("The True Negatives and False Negatives Rates are:")
print(kmc.NegativesRate(classification, assignments, hemoglobin))
print("The True Posatives and False Posgatives Rates are:")
print(kmc.PosativesRate(classification, assignments, hemoglobin))