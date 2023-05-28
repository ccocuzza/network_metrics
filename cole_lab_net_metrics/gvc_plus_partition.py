# C. Cocuzza, 2022. Based on gvcAlgorithm.m, Cole et al., 2013.
# global variability coefficient (GVC), a measure of flexible hubs.
# The module gvc.py has the core function to estimate GVC, this version 
# has additional code to sort into functional network communities 
# and return results at the network level (see input/output notes below).

import numpy as np

def gvc(fcArray,netBoundaries,nodeOrder,useRestPartitionAdjuster=False,netBoundariesNew=None,nodeOrderNew=None):
    '''
        INPUTS:
    1. fcArray: a connectivity array of size: nodes x nodes x task conditions x subjects; do NOT sort to partition beforehand (this is handled by nodeOrder; plus possibly nodeOrderNew; see notes on these inputs below).
    2. netBoundaries: a helper variable that specifies information about the resting-state partition you'd like to use (Cocuzza et al., 2020 used the Cole Anticevic brain wide network partition, or CAB-NP; the helper variable included in this package boundariesCA.npy can be used here). It is of size: number of networks x 3. 1st column = start region index of network; 2nd column = end region index of network; 3rd column = network size. For example in te CAB-NP, VIS1 (or primary visual network) starts at region 0 and ends at region 5, so the first row of boundariesCA is: [0 5 6].
    3. nodeOrder: This is an indexing vector of size: number of nodes (should match first and second dimensions of fcArray). The values in this vector will "sort" the fcArray to the resting-state partition of interest (i.e., re-index). A helper variable is included in this packaged called nodeOrder.npy that can be used here. 
    4. useRestPartitionAdjuster: optional. default is False; if True 
    5. netBoundariesNew: optional. default is None; if setting useRestPartitionAdjuster to True, this should be array of the same format as netBoundaries, but adjusted based on empirical resting-state data (see Cocuzza et al., 2020 "empirically adjusted CABNP" in the Methods); also see helper function restPartitionAdjuster.py 
    6. nodeOrderNew: optional. default is None; if setting useRestPartitionAdjuster to True, this should be an array of the same format as nodeOrder, but adjusted based on empirical resting-state data (see Cocuzza et al., 2020 "empirically adjusted CABNP" in the Methods); also see helper function restPartitionAdjuster.py 
    
    OUTPUTS:
    gvcNodesSubjs: an array of GVC scores of size: nodes x subjects. Each parent (row) node from fcArray has a GVC score, which per target node, is the standard deviation of each of it's source (column) nodes connectivity weights across task conditions, which is then averaged across source nodes to get 1 score. the same result as gvc.py
    gvcNetsSubjs: an array of GVC scores of size: networks x subjects. Same as gvcNodesSubjs, but regions (nodes) are clustered into their corresponding functional networks. Each row contains mean GVC scores of all the regions in that network (per subject).
    gvcNodes: a vector of GVC scores of size: number of nodes. This is the grand mean GVC for each region (i.e., just the mean across subjects of gvcNodesSubjs).
    '''
    
    if useRestPartitionAdjuster:
        boundariesHere = netBoundariesNew.copy()
    elif not useRestPartitionAdjuster: 
        boundariesHere = netBoundaries.copy()
        
    numNets = boundariesHere.shape[0]

    nParcels = fcArray.shape[0]
    numTasks = fcArray.shape[2]
    numSubjs = fcArray.shape[3]
    
    # Compute GVC 
    gvcNodesSubjs = np.zeros((nParcels,numSubjs))
    tempSubj = np.zeros((nParcels,nParcels,numTasks))
    for subjNum in range(numSubjs):
        thisSubj = fcArray[:,:,:,subjNum]
        for taskNum in range(numTasks): # the purpose of this loop is just to fill in the diagonals with NaNs (may be a better way to do this)
            thisTask = thisSubj[:,:,taskNum]
            np.fill_diagonal(thisTask,np.nan)
            tempSubj[:,:,taskNum] = thisTask

        gvcNodesSubjs[:,subjNum] = np.nanmean(np.nanstd(tempSubj,axis=2),axis=1) # variability across states --> mean of connecting nodes 

    if useRestPartitionAdjuster:
        gvcNodesSubjs = gvcNodesSubjs[nodeOrder,:][nodeOrderNew,:]
    elif not useRestPartitionAdjuster: 
        gvcNodesSubjs = gvcNodesSubjs[nodeOrder,:]

    gvcNodes = np.nanmean(gvcNodesSubjs,axis=1)

    # Cluster into networks 
    gvcNetsSubjs = np.zeros((numNets,numSubjs))
    for subjNum in range(numSubjs):
        thisSubj = gvcNodesSubjs[:,subjNum]
        for netNum in range(numNets):
            startNode = boundariesHere[netNum,0].astype(int)
            endNode = (boundariesHere[netNum,1]+1).astype(int)
            thisCluster = thisSubj[startNode:endNode]
            gvcNetsSubjs[netNum,subjNum] = np.nanmean(thisCluster)
            
    return gvcNodesSubjs, gvcNetsSubjs, gvcNodes
