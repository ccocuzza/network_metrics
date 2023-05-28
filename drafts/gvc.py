# C. Cocuzza, 2022. Based on gvcAlgorithm.m, Cole et al., 2013.
# global variability coefficient (GVC), a measure of flexible hubs.

import numpy as np

def gvc(fcArray):
    '''
    INPUTS:
    fcArray: a connectivity array of size: nodes x nodes x task conditions x subjects 
    
    OUTPUTS:
    gvcNodesSubjs: an array of GVC scores of size: nodes x subjects. Each parent (row) node from fcArray has a GVC score, which per target node, is the standard deviation of each of it's source (column) nodes connectivity weights across task conditions, which is then averaged across source nodes to get 1 score. 
    
    '''
    
    nParcels = fcArray.shape[0]
    numTasks = fcArray.shape[2]
    numSubjs = fcArray.shape[3]
    
    # Compute GVC 
    gvcNodesSubjs = np.zeros((nParcels,numSubjs))
    tempSubj = np.zeros((nParcels,nParcels,numTasks))
    for subjNum in range(numSubjs):
        thisSubj = fcArray[:,:,:,subjNum]
        for taskNum in range(numTasks): 
            thisTask = thisSubj[:,:,taskNum]
            np.fill_diagonal(thisTask,np.nan)
            tempSubj[:,:,taskNum] = thisTask

        gvcNodesSubjs[:,subjNum] = np.nanmean(np.nanstd(tempSubj,axis=2),axis=1) # variability across states --> mean of connecting nodes 
            
    return gvcNodesSubjs