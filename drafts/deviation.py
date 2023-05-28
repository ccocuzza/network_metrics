# C. Cocuzza, 2022. Based on Cocuzza et al., 2020, J Neurosci.

import numpy as np

def deviation(fcArray,netBoundaries,nodeOrder,useMeanFirst=False,useRestPartitionAdjuster=False,netBoundariesNew=None,nodeOrderNew=None):

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
        nodePrefValsMeanFirst = np.zeros((nParcels,numTasks))
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
            nodePrefValsMeanFirst[:,ruleSet] = tempNodeVecPrefs
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
        nodePrefValsMeanFirst = np.zeros((nParcels,numTasks,numSubjs))
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
                nodePrefValsMeanFirst[:,taskNum,subjNum] = tempNodeVecPrefs
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
        
    return deviationRFs, clusteredAffinities, netAffinities, nodePrefValsMeanFirst, nodePrefIdxs, maxMembershipsTF, maxMembershipsMean