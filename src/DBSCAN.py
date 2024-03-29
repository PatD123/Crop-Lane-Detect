import numpy as np

class DBSCAN:
    def __init__(self, eps, minPts):
        self.dbscan_dict = {}
        self.points = []
        self.minPts = minPts
        self.eps = eps

    def update(self, line):
        line = line.squeeze()
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        self.dbscan_dict[(mid_x, mid_y)] = line
        self.points.append(np.array([mid_x, mid_y]))

    def scan(self):
        '''
        Cluster the dataset `D` using the DBSCAN algorithm.
        
        dbscan takes a dataset `D` (a list of vectors), a threshold distance
        `eps`, and a required number of points `MinPts`.
        
        It will return a list of cluster labels. The label -1 means noise, and then
        the clusters are numbered starting from 1.
        '''

        labels = [0]*len(self.points)

        # C is the ID of the current cluster.    
        C = 0
        
        # For each point P in the Dataset D...
        # ('P' is the index of the datapoint, rather than the datapoint itself.)
        for P in range(0, len(self.points)):
        
            # Only points that have not already been claimed can be picked as new 
            # seed points.    
            # If the point's label is not 0, continue to the next point.
            if not (labels[P] == 0):
                continue
            
            # Find all of P's neighboring points.
            NeighborPts = self.region_query(P)
            
            # If the number is below MinPts, this point is noise. 
            # This is the only condition under which a point is labeled 
            # NOISE--when it's not a valid seed point. A NOISE point may later 
            # be picked up by another cluster as a boundary point (this is the only
            # condition under which a cluster label can change--from NOISE to 
            # something else).
            if len(NeighborPts) < self.minPts:
                labels[P] = -1
            # Otherwise, if there are at least MinPts nearby, use this point as the 
            # seed for a new cluster.    
            else: 
                C += 1
                self.grow_cluster(labels, P, NeighborPts, C)
        
        # All data has been clustered!
        return labels


    def grow_cluster(self, labels, P, NeighborPts, C):
        '''
        Grow a new cluster with label `C` from the seed point `P`.
        
        This function searches through the dataset to find all points that belong
        to this new cluster. When this function returns, cluster `C` is complete.
        
        Parameters:
        `D`      - The dataset (a list of vectors)
        `labels` - List storing the cluster labels for all dataset points
        `P`      - Index of the seed point for this new cluster
        `NeighborPts` - All of the neighbors of `P`
        `C`      - The label for this new cluster.  
        `eps`    - Threshold distance
        `MinPts` - Minimum required number of neighbors
        '''

        # Assign the cluster label to the seed point.
        labels[P] = C
        
        # FIFO
        i = 0
        while i < len(NeighborPts):    
            
            # Get the next point from the queue.        
            Pn = NeighborPts[i]
        
            # If Pn was labelled NOISE during the seed search, then we
            # know it's not a branch point (it doesn't have enough neighbors), so
            # make it a leaf point of cluster C and move on.
            if labels[Pn] == -1:
                labels[Pn] = C
            
            # Otherwise, if Pn isn't already claimed, claim it as part of C.
            elif labels[Pn] == 0:
                # Add Pn to cluster C (Assign cluster label C).
                labels[Pn] = C
                
                # Find all the neighbors of Pn
                PnNeighborPts = self.region_query(Pn)
                
                # If Pn has at least MinPts neighbors, it's a branch point!
                # Add all of its neighbors to the FIFO queue to be searched. 
                if len(PnNeighborPts) >= self.minPts:
                    NeighborPts = NeighborPts + PnNeighborPts
                # If Pn *doesn't* have enough neighbors, then it's a leaf point.
                # Don't queue up it's neighbors as expansion points.
                #else:
                    # Do nothing                
                    #NeighborPts = NeighborPts               
            
            # Advance to the next point in the FIFO queue.
            i += 1        
        
        # We've finished growing cluster C!


    def region_query(self, P):
        '''
        Find all points in dataset `D` within distance `eps` of point `P`.
        
        This function calculates the distance between a point P and every other 
        point in the dataset, and then returns only those points which are within a
        threshold distance `eps`.
        '''
        neighbors = []
        
        # For each point in the dataset...
        for Pn in range(0, len(self.points)):
            
            # If the distance is below the threshold, add it to the neighbors list.
            if np.linalg.norm(self.points[P] - self.points[Pn]) < self.eps:
                neighbors.append(Pn)
        return neighbors
    
    def return_max(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        if len(values) == 0 or (values[0] == -1 and len(values) == 1):
            return []
        idx = np.argmax(counts)
        if values[idx] == -1:
            idx = np.argmax(counts[1:]) + 1

        lines = []
        for i in range(len(labels)):
            if(labels[i] == values[idx]):
                lines.append(self.dbscan_dict[(self.points[i][0], self.points[i][1])])
        return lines

