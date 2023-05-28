# C. Cocuzza, 2022. Based on Cocuzza et al., 2020, J Neurosci.

import numpy as np
import scipy.stats as stats

def restPartitionAdjuster(fcArray,netBoundaries,nodeOrder):
    '''
    INPUTS:
    1. fcArray: a resting-state (or otherwise intrinsic) connectivity array of size: nodes x nodes x subjects; do NOT sort to partition beforehand (this is handled by nodeOrder)
    2. netBoundaries: a helper variable that specifies information about the original resting-state partition that is to be adjusted (Cocuzza et al., 2020 used the Cole Anticevic brain wide network partition, or CAB-NP (Ji et al., 2019); the helper variable included in this package boundariesCA.npy can be used here). It is of size: number of networks x 3. 1st column = start region index of network; 2nd column = end region index of network; 3rd column = network size. For example in the CAB-NP, VIS1 (or primary visual network) starts at region 0 and ends at region 5, so the first row of boundariesCA is: [0 5 6].
    3. nodeOrder: This is an indexing vector of size: number of nodes (should match first and second dimensions of fcArray). The values in this vector will "sort" the fcArray to the original resting-state partition of interest (i.e., re-index). A helper variable is included in this package called nodeOrder.npy that can be used here. 
    
    OUTPUTS:
    1. netBoundariesNew: an array of size: networks x 3. This is the same format as netBoundaries, but with the adjustment made based on empirical resting-state FC estimates. This can be used in deviation.py and gvc_plus_partition.py.
    2. nodeOrderNew: an array of size: number of nodes (ie regions). This is the same format as nodeOrder, but with the adjustment made based on empirical resting-state FC estimates. This can be used in deviation.py and gvc_plus_partition.py.
    3. restPreferences: an array of size: nodes x subjects. For each region (node) and subject, the number indicates network index that is preferred. 
    4. percentAgree: an array of size: number of nodes (ie regions). For each region (node) this is the percent (in decimal format) of subjects that lead to the consensus. 
    5. restConsensus: an array of size: number of nodes (ie regions). For each region (node) this is the preferred network index (cross-subject consensus with percentAgree agreement).
    6. nodeIndicesNew: an array of size: number of nodes (ie regions). This is the adjusted network affiliation vector (each region's network number).
    '''
    
    nParcels = fcArray.shape[0]
    numSubjs = fcArray.shape[2]
    
    numNets = netBoundaries.shape[0]
    
    restPreferences = np.zeros((nParcels,numSubjs))
    for subjNum in range(numSubjs):
        thisSubjRestFC = fcArray[:,:,subjNum].copy()
        subjSortedCA = thisSubjRestFC[nodeOrder,:][:,nodeOrder]
        np.fill_diagonal(subjSortedCA,np.nan)
        
        for nodeNum in range(nParcels):
            thisNodeVec = subjSortedCA[nodeNum,:]
            tempVec = np.zeros((numNets))
            
            for netNum in range(numNets):
                startNode = netBoundaries[netNum,0].astype(int)
                endNode = (netBoundaries[netNum,1]+1).astype(int)
                tempVec[netNum] = np.tanh(np.nanmean(np.arctanh(thisNodeVec[startNode:endNode])))
                
            prefIdx = np.argsort(tempVec)[::-1]
            prefVals = np.sort(tempVec)[::-1] # 0 index should equal np.max(tempVec); note [::-1] puts it in descending order (flips array)
            restPreferences[nodeNum,subjNum] = prefIdx[0]

    # find consensus: if 50% or more of subjects have this preference 
    restMode, restModeCount = stats.mode(restPreferences,axis=1)
    percentAgree = np.zeros((nParcels))
    for nodeNum in range(nParcels):
        thisMode = restMode[nodeNum].astype(int)[0]
        percentAgree[nodeNum] = (restModeCount[nodeNum].astype(int)[0])/numSubjs

    restConsensus = np.zeros((nParcels))
    for netNum in range(numNets):
        startNode = netBoundaries[netNum,0].astype(int)
        endNode = (netBoundaries[netNum,1]+1).astype(int)
        for nodeNum in range(startNode,endNode):
            thisNode = percentAgree[nodeNum]
            if thisNode<0.5:
                restConsensus[nodeNum] = netNum
            elif thisNode>=0.5:
                restConsensus[nodeNum] = restMode[nodeNum]

    # generate netBoundariesNew array for use in other cells (to replace netBoundaries)
    nodeIndicesNew = np.sort(restConsensus)
    nodeOrderNew = np.argsort(restConsensus)
    netBoundariesNew = np.zeros((numNets,3))
    for netNum in range(numNets):
        finder = np.where(nodeIndicesNew==netNum)[0]
        vals = [np.min(finder),np.max(finder)]
        valVec = np.arange(np.min(finder),np.max(finder)+1)
        sampSize = valVec.shape[0]
        netBoundariesNew[netNum,0:2] = vals
        netBoundariesNew[netNum,2] = sampSize
        
    return netBoundariesNew, nodeOrderNew, restPreferences, percentAgree, restConsensus, nodeIndicesNew