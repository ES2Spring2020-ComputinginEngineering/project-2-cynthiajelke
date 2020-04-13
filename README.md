This project is based on an example and dataset from Data Science course developed at Berkeley (Data8.org).

    Cynthia Jelke 
    Project 2

ckd.csv

    contains the hemoglobin, glucose, and classification values for each data point
    
  
NearestNeighborClassification

    the code for Nearest Neighbor Classification. Classifies a random data-point as part of class 0 or 1 based on its
    nearest neighbors' classifications.
    
    How to use:
    1. scroll to the bottom where it says "MAIN SCRIPT"
    2. change k to equal however many neighbors you want to compare the point to
    3. if you want to run nearestNeighborClassifier instead of K_nearestClassification, uncomment line 109 and comment
       line 106
    4. click run (the new point will be graphed in the shape of a diamond)
  
KMeansClustering_functions

    contains the functions required to run K Means Clustering algorithm on the data points.
    
    Functions:
    1. openckdfile()
          Parameters: none
          Purpose: reads files as arrays for glucose, hemoglobin, and their classification and normalizes the values to they’re on the
          same scale
          Output: glucose, hemoglobin, and classification arrays
    2. select(K)
          Parameters: K - how many centroid you want
          Purpose: chooses k number of random centroids
          Output: an array of k centroid points
    3. assign(centroids, hemoglobin, glucose)
          Parameters: centroids - an array of centroids fo the clusters
                      hemoglobin - an array of the hemoglobin values
                      glucose - an array of the glucose values
          Purpose: sets the data points as part of clusters based on distance from centroids
          Output: an array of assigned classifications based on the centroid points
    4. findClusterMean(assignments, hemoglobin, glucose, clusternum) 
          Parameters: assignments - the classifications assigned to each point
                      hemoglobin - an array of the hemoglobin values
                      glucose - an array of the glucose values
                      clusternum - which cluster we are looking at (a value 0 through k-1)
          Purpose: finds the mean point (center) of a cluster
          Output: the center value of a cluster (aka its new centroid)
    5. kMeansClustering(k, iteration_count)
          Purpose: applies above functions to properly use K Means Clustering to assign clusters
          Output: the final assignments as to which groups the points belong to
    6. graphingKMeans(glucose, hemoglobin, assignment, centroids, k)
          Parameters: assignment - the classifications assigned to each point
                      centroids - an array of centroids fo the clusters
                      hemoglobin - an array of the hemoglobin values
                      glucose - an array of the glucose values
                      k - how many clusters you want
          Purpose: graphs the data points in their assign clusters
    7. NegativesRate(classification, assignments, hemoglobin)
          Purpose: calculates the True Negative Rate and the False Negative Rate
          Output: the true negative rate and the false negative rate
    8. PositivesRate(classification, assignments, hemoglobin)
          Purpose: calculates the True Positive Rate and the False Positive Rate
          Output: the true positive rate and the false positive rate

  
KMeansClustering_driver

    the code for running K Means Clustering algorithm. Displays a scatterplot of the assigned clusters and prints the
    centroid values, true/false positives rate, and true/false negatives rate.
    
    How to use:
    1. change k to be however many clusters you want it sorted into (results are weird after k=5 because of data)
    2. if you want to change the iteration count change the second parameter in "kmc.kMeansClustering" on line 19
    3. run the code 
     
     
