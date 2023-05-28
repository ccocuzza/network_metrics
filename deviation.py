# C. Cocuzza, 2022. Based on Cocuzza et al., 2020, J Neurosci.
# Network partition deviation, or just "deviation", measures the relative frequency 
# that a given node's connectivity patterns deviate from their resting-state derived partition 
# across task conditions.

# EX: deviation score of 0.5 for regio 1 indicates that region 1's connectivity deviated from
# it's resting-state partition across 50% of task states.

# In Cocuzza et al., 2020, regions in the cingulo-opercular network (CON) (a cognitive control network) 
# exhibited particularly high deviation, indicating that CON was reconfiguring (from its 
# intrinsic configuration) with a high rate across task states, possibly to lend task-relevant processing resources 
# to other networks (see various affinities results to assess which networks a given region reassigns to)

# Main output = deviationRFs containing deviation scores; see output list for other potentially useful results that are returned.
# Also see input list notes below for tips and useful helper arrays included in this package.

import numpy as np

def deviation(fcArray,netBoundaries,nodeOrder,useMeanFirst=False,useRestPartitionAdjuster=False,netBoundariesNew=None,nodeOrderNew=None):
    '''
    INPUTS:
    1. fcArray: a connectivity array of size: nodes x nodes x task conditions x subjects; do NOT sort to partition beforehand (this is handled by nodeOrder; plus possibly nodeOrderNew; see notes on these inputs below).
    2. netBoundaries: a helper variable that specifies information about the resting-state partition you'd like to use (Cocuzza et al., 2020 used the Cole Anticevic brain wide network partition, or CAB-NP (Ji et al., 2019); the helper variable included in this package boundariesCA.npy can be used here). It is of size: number of networks x 3. 1st column = start region index of network; 2nd column = end region index of network; 3rd column = network size. For example in the CAB-NP, VIS1 (or primary visual network) starts at region 0 and ends at region 5, so the first row of boundariesCA is: [0 5 6].
    3. nodeOrder: This is an indexing vector of size: number of nodes (should match first and second dimensions of fcArray). The values in this vector will "sort" the fcArray to the resting-state partition of interest (i.e., re-index). A helper variable is included in this package called nodeOrder.npy that can be used here. 
    4. useMeanFirst: optional. default is False; if True, this will take the mean across subjects first and all results will be in aggregate form (if False results will be returned for each subject)
    5. useRestPartitionAdjuster: optional. default is False; if True this will adjust the resting-state partition to empirical resting-state connectivity estimates(see Cocuzza et al., 2020 "empirically adjusted CABNP" in the Methods); also see helper function restPartitionAdjuster.py.
    6. netBoundariesNew: optional. default is None; if setting useRestPartitionAdjuster to True, this should be an array of the same format as netBoundaries, but adjusted based on empirical resting-state data (see Cocuzza et al., 2020 "empirically adjusted CABNP" in the Methods); also see helper function restPartitionAdjuster.py 
    7. nodeOrderNew: optional. default is None; if setting useRestPartitionAdjuster to True, this should be an array of the same format as nodeOrder, but adjusted based on empirical resting-state data (see Cocuzza et al., 2020 "empirically adjusted CABNP" in the Methods); also see helper function restPartitionAdjuster.py 
    
    OUTPUTS:
    1. deviationRFs: an array of shape: number of networks x subjects. This is the main deviation result. Note: you can multiply by 100 to put it into percent form (i.e., percent of task states where a given node's connectivity deviated from the resting-state configuration). Note: 1-deviationRFs can give the complementary score of 'adherence', or how often (across task states) the resting-state configuration was adhered to.
    2. clusteredAffinities: an array of shape: networks x networks x subjects. This will give the frequency (across task states) that a given network preferred another network (so diagonal = adherence; sum of off-diagonals = deviation; rows should each sum to 1)
    3. netAffinities: an array of shape: nodes x networks x subjects. for each region (node) and subject (e.g., netAffinities[region_index,:,subject_index]), a vector (of size: number of networks) of how often (how many task states) that region preferred a given network (i.e., clusteredAffinites at the region level).
    4. nodePrefVal: an array of size: nodes x task conditions. These are the maximum edge weights used to determine preferred connection per task condition (see Cocuzza et al., 2020). 
    5. nodePrefIdxs: an array of size: nodes x task conditions. If a region (node) deviates from it's resting-state configuration, this gives the network number that it reassigned to (for each task state)
    6. maxMembershipsTF: an array of size: nodes x task conditions x networks x subjects. This is the full binarized array used to determine deviation scores. For a given region, subject, and task state (e.g., maxMembershipsTF[region_index,task_index,:,subject_index] which is a vector of size: number of networks) a 1 will will be in the prefferred network index. This relates to the output netAffinities in the following way: np.array_equal(netAffinities,(np.sum(maxMembershipsTF,axis=1) / number_conditions))
    '''
    if useRestPartitionAdjuster:
        boundariesHere = netBoundariesNew.copy()
    elif not useRestPartitionAdjuster: 
        boundariesHere = netBoundaries.copy()
        
    numNets = boundariesHere.shape[0]

    nParcels = fcArray.shape[0]
    numTasks = fcArray.shape[2]
    numSubjs = fcArray.shape[3]
        
    if useMeanFirst:
        # Cluster FC vals down to network-level (weighted avg) --> [node x network x task FC matrices]
        print('Clustering FC values down to the network level (weighted average) --> node x network x task FC matrices...')
        clusteredTaskFC = np.zeros((nParcels,numNets,numTasks))
        tempNet = np.zeros((numNets))
        tempTask = np.zeros((nParcels,numNets))
        for ruleSet in range(numTasks):
            if useRestPartitionAdjuster:
                if ruleSet==0: 
                    print('Task FC: estimated with combinedFC & nodes sorted/clustered per empirically-adjusted CAB-NP')
                thisTaskHere = fcArray[:,:,ruleSet]
                thisTask = thisTaskHere[nodeOrder,:][:,nodeOrder][nodeOrderNew,:][:,nodeOrderNew]
            elif not useRestPartitionAdjuster:
                if ruleSet==0: 
                    print('Task FC: estimated with combinedFC & nodes sorted/clustered per original CAB-NP')
                thisTaskHere = fcArray[:,:,ruleSet]
                thisTask = thisTaskHere[nodeOrder,:][:,nodeOrder]
            for nodeNum in range(nParcels):
                thisNodeVec = thisTask[nodeNum,:]
                for netNum in range(numNets):
                    startNode = boundariesHere[netNum,0].astype(int)
                    endNode = (boundariesHere[netNum,1]+1).astype(int)
                    thisCluster = thisNodeVec[startNode:endNode]
                    clusterMean = np.nanmean(thisCluster)
                    tempNet[netNum] = clusterMean
                tempTask[nodeNum,:] = tempNet
            clusteredTaskFC[:,:,ruleSet] = tempTask

        # Find max FC values and indices (e.g., connecting node number) of net means from above --> [node x task preference matrix]
        print('Finding max FC values and their indices (e.g., connecting node number) --> node x task preference matrices...');
        tempNodeVecPrefs = np.zeros((nParcels))
        nodePrefVals = np.zeros((nParcels,numTasks))
        tempNodeVecIdxs = np.zeros((nParcels))
        nodePrefIdxs = np.zeros((nParcels,numTasks))
        for ruleSet in range(numTasks):
            thisTask = clusteredTaskFC[:,:,ruleSet]
            for nodeNum in range(nParcels):
                thisNodeVec = thisTask[nodeNum,:]
                maxVal = np.max(thisNodeVec)
                maxIdx = np.argmax(thisNodeVec)
                tempNodeVecPrefs[nodeNum] = maxVal
                tempNodeVecIdxs[nodeNum] = maxIdx
            nodePrefVals[:,ruleSet] = tempNodeVecPrefs
            nodePrefIdxs[:,ruleSet] = tempNodeVecIdxs

        # Iterating over all networks: test if preference index is a member of that network --> [node x task x network binarized affinity matrix]
        print('Iterating over all networks to test if the preference '+
              'index (ie max from above) is a member of that network --> node x task x network binarized affinity matrices...')
        maxMembershipsTF = np.zeros((nParcels,numTasks,numNets))
        maxMembershipsMean = np.zeros((nParcels,numTasks,numNets))
        for nodeNum in range(nParcels):
            thisNodeVec = nodePrefIdxs[nodeNum,:]
            for netNum in range(numNets):
                memberTF = (thisNodeVec==netNum)*1
                maxMembershipsTF[nodeNum,:,netNum] = memberTF # kept binary
                maxMembershipsMean[nodeNum,:,netNum] = memberTF * netNum # to get netIdx

        # Find affinity scores (relative frequencies or RFs) for all connecting networks --> [node x network relative freq NPA values]
        print('Finding affinity scores (relative frequencies) for all connecting networks --> node x network x relative frequency NPA matrices...')
        netAffinities = np.zeros((nParcels,numNets))
        for netNum in range(numNets):
            thisNet = maxMembershipsTF[:,:,netNum]
            for nodeNum in range(nParcels):
                thisNodeVec = thisNet[nodeNum,:]
                thisNodeSum = np.sum(thisNodeVec)
                thisNodeRF = thisNodeSum / numTasks # Tally --> RF
                netAffinities[nodeNum,netNum] = thisNodeRF # Each row should add to 1 with np.sum(netAffinities,axis=1)

        # Cluster ROIs into NOIs --> [network x network relative frequency NPA values]
        print('Clustering ROIs into NOIs --> network x network relative frequency NPA matrices...')
        clusteredAffinities = np.zeros((numNets,numNets))
        for netNum in range(numNets):
            startNode = boundariesHere[netNum,0].astype(int)
            endNode = (boundariesHere[netNum,1]+1).astype(int)
            thisCluster = netAffinities[startNode:endNode,:]
            clusterMeanRFs = np.nanmean(thisCluster,axis=0) # should still add to 1 per row
            clusteredAffinities[netNum,:] = clusterMeanRFs

        # Sanity check: they all add to 100% 
        print('Sanity check that all clustered affinities add to 100%...');
        for netNum in range(numNets):
            thisNet = clusteredAffinities[netNum,:]
            fullPercentTest = (np.sum(thisNet)) * 100
            if fullPercentTest<99.9:
                print('NOI # ' + str(netNum) + ' adds to ' + str(fullPercentTest) + '%; not the expected ~100%. Please check.')

        # Pulling out adherence to pre-defined network partition --> [network x 1] NPA vector 
        print('Pulling out adherence to pre-defined network partition --> network NPA matrices...')
        #adherenceRFs = np.zeros((numNets))
        adherenceRFs= np.diag(clusteredAffinities)

        # Calculating deviation (NPD) as 1-NPA --> [network x 1] NPD vector 
        print('Computing deviation (NPD) --> network NPD matrices...');
        deviationRFs = 1 - adherenceRFs

    ##################################################################################################################
    elif not useMeanFirst:
        # Cluster FC vals down to network-level (weighted avg) --> [node x network x task FC matrices]
        print('Clustering FC values down to the network level (weighted average) --> node x network x task x subject FC matrices...')
        clusteredTaskFC = np.zeros((nParcels,numNets,numTasks,numSubjs))
        tempNet = np.zeros((numNets))
        tempTask = np.zeros((nParcels,numNets))
        for subjNum in range(numSubjs):
            for taskNum in range(numTasks):
                if useRestPartitionAdjuster:
                    if taskNum==0 and subjNum==0: 
                        print('Task FC: estimated with combinedFC & nodes sorted/clustered per empirically-adjusted CAB-NP')
                    thisTaskHere = fcArray[:,:,taskNum,subjNum]
                    thisTask = thisTaskHere[nodeOrder,:][:,nodeOrder][nodeOrderNew,:][:,nodeOrderNew]
                elif not useRestPartitionAdjuster:
                    if taskNum==0 and subjNum==0: 
                        print('Task FC: estimated with combinedFC & nodes sorted/clustered per original CAB-NP')
                    thisTaskHere = fcArray[:,:,taskNum,subjNum]
                    thisTask = thisTaskHere[nodeOrder,:][:,nodeOrder]
                for nodeNum in range(nParcels):
                    thisNodeVec = thisTask[nodeNum,:]
                    for netNum in range(numNets):
                        startNode = boundariesHere[netNum,0].astype(int)
                        endNode = (boundariesHere[netNum,1]+1).astype(int)
                        thisCluster = thisNodeVec[startNode:endNode]
                        clusterMean = np.nanmean(thisCluster)
                        tempNet[netNum] = clusterMean
                    tempTask[nodeNum,:] = tempNet
                clusteredTaskFC[:,:,taskNum,subjNum] = tempTask

        # Find max FC values and indices (e.g., connecting node number) of net means from above --> [node x task preference matrix]
        print('Finding max FC values and their indices (e.g., connecting node number) --> node x task x subject preference matrices...')
        tempNodeVecPrefs = np.zeros((nParcels))
        nodePrefVals = np.zeros((nParcels,numTasks,numSubjs))
        tempNodeVecIdxs = np.zeros((nParcels))
        nodePrefIdxs = np.zeros((nParcels,numTasks,numSubjs))
        for subjNum in range(numSubjs):
            for taskNum in range(numTasks):
                thisTask = clusteredTaskFC[:,:,taskNum,subjNum]
                for nodeNum in range(nParcels):
                    thisNodeVec = thisTask[nodeNum,:]
                    maxVal = np.max(thisNodeVec)
                    maxIdx = np.argmax(thisNodeVec)
                    tempNodeVecPrefs[nodeNum] = maxVal
                    tempNodeVecIdxs[nodeNum] = maxIdx
                nodePrefVals[:,taskNum,subjNum] = tempNodeVecPrefs
                nodePrefIdxs[:,taskNum,subjNum] = tempNodeVecIdxs

        # Iterating over all networks: test if preference index is a member of that network --> [node x task x network binarized affinity matrix]
        print('Iterating over all networks to test if the preference index '+
              '(ie max from above) is a member of that network --> node x task x network x subject binarized affinity matrices...')
        maxMembershipsTF = np.zeros((nParcels,numTasks,numNets,numSubjs))
        maxMembershipsMean = np.zeros((nParcels,numTasks,numNets,numSubjs))
        for subjNum in range(numSubjs):
            for nodeNum in range(nParcels):
                thisNodeVec = nodePrefIdxs[nodeNum,:,subjNum]
                for netNum in range(numNets):
                    memberTF = (thisNodeVec==netNum)*1
                    maxMembershipsTF[nodeNum,:,netNum,subjNum] = memberTF # kept binary
                    maxMembershipsMean[nodeNum,:,netNum,subjNum] = memberTF * netNum # to get netIdx

        # Find affinity scores (relative frequencies or RFs) for all connecting networks --> [node x network relative freq NPA values]
        print('Finding affinity scores (relative frequencies) for all connecting networks --> node x network x relative frequency NPA matrices...')
        netAffinities = np.zeros((nParcels,numNets,numSubjs))
        for subjNum in range(numSubjs):
            for netNum in range(numNets):
                thisNet = maxMembershipsTF[:,:,netNum,subjNum]
                for nodeNum in range(nParcels):
                    thisNodeVec = thisNet[nodeNum,:]
                    thisNodeSum = np.sum(thisNodeVec)
                    thisNodeRF = thisNodeSum / numTasks # Tally --> RF
                    netAffinities[nodeNum,netNum,subjNum] = thisNodeRF # Each row should add to 1 with np.sum(netAffinities,axis=1)

        # Cluster ROIs into NOIs --> [network x network relative frequency NPA values]
        print('Clustering ROIs into NOIs --> network x network x subject relative frequency NPA matrices...')
        clusteredAffinities = np.zeros((numNets,numNets,numSubjs))
        for subjNum in range(numSubjs):
            thisSubj = netAffinities[:,:,subjNum]
            for netNum in range(numNets):
                startNode = boundariesHere[netNum,0].astype(int)
                endNode = (boundariesHere[netNum,1]+1).astype(int)
                thisCluster = thisSubj[startNode:endNode,:]
                clusterMeanRFs = np.nanmean(thisCluster,axis=0) # should still add to 1 per row
                clusteredAffinities[netNum,:,subjNum] = clusterMeanRFs 

        # Sanity check: they all add to 100% 
        print('Sanity check that all clustered affinities add to 100%...');
        for subjNum in range(numSubjs):
            for netNum in range(numNets):
                thisNet = clusteredAffinities[netNum,:,subjNum]
                fullPercentTest = (np.sum(thisNet)) * 100
                if fullPercentTest<99.9:
                    print('SUBJ' + str(subjNum) + ', NOI # ' + str(netNum) + ' adds to ' + str(fullPercentTest) + '%; not the expected ~100%. Please check.')

        # Pulling out adherence to pre-defined network partition --> [network x 1] NPA vector 
        print('Pulling out adherence to pre-defined network partition --> network x subject NPA matrices...')
        adherenceRFs = np.zeros((numNets,numSubjs))
        for subjNum in range(numSubjs):
            adherenceRFs[:,subjNum] = np.diag(clusteredAffinities[:,:,subjNum])

        # Calculating deviation (NPD) as 1-NPA --> [network x 1] NPD vector 
        print('Computing deviation (NPD) --> network x subject NPD matrices...')
        deviationRFs = 1 - adherenceRFs
        
    return deviationRFs, clusteredAffinities, netAffinities, nodePrefVals, nodePrefIdxs, maxMembershipsTF