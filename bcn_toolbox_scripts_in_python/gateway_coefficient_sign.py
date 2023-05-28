# C. Cocuzza, 2022. Compute gateway coefficicent (GC): converting brain connectivity toolbox code (MATLAB) to python.
# Based on code here: https://drive.google.com/drive/folders/1eTYY4NAxMo7L2SLSyAUk8bI1JUHTIx7l
# See Vargas ER, Wahl LM. Eur Phys J B (2014) 87:1-10.
# Gateway is similar to participation coefficient, but will weight how critical connections are to intermodular
# connectivity. EX: if a node is the only connections between its network and another network,
# it will have a higher gateway score 
# For connectivity arrays that are weighted and include positive/negative (signed) values.

# NOTE: a helper variable is included in this package called thisNetAffilVec.npy that can be used for the input thisNetAffilVec. This is based on the Cole Anticevic brain wide network partition (CABNP) (see Ji et al., 2019 or https://github.com/ColeLab/ColeAnticevicNetPartition). In addition, network names (in proper order 1-12, or 0-11 in python indexing) can be found in networkNamesCABNP_Long.npy (and corresponding acronyms in netNamesCABNP_Short.npy).

import numpy as np 

def gateway_coefficient_sign(fcArray,thisNetAffilVec,numNets,centType='node_strength'):
    '''
    # INPUTS:
    # fcArray = a connectivity array of size: nodes x nodes x task conditions; either as-is or adjusted for pos/neg before hand (i.e., can mask for positive values, then run the function for just positive weights to get a positive DC); also sorted to network partition beforehand. NOTE: the recommended use is to pre-index fcArray over subjects and iteratively call this module, for example using fcArray[:,:,:,subject_index] as the fcArray input.
    # thisNetAffilVec = m, network assignment (i.e., affiliation) vector (note: start numbering at 1); this sorting should match fcArray
    # numNets = number of networks in thisNetAffilVec; should equal largest integer in that thisNetAffilVec
    # centType: a string either 'node_strength' (default) or 'betweenness'; which centrality measure to use 

    # OUTPUTS: 
    # partCoef = m, vector of participation scores for each parent node 
    '''
    
    nVertsInGraph = int(len(fcArray))
    np.fill_diagonal(fcArray,0) # zero out diagonal
    degreeOfROIs = np.nansum(fcArray,axis=1)
    netAffilMask = np.matmul((fcArray!=0),(np.diag(thisNetAffilVec)))
    nModules = int(np.nanmax(thisNetAffilVec))

    ksArr = np.zeros((nVertsInGraph,nModules))
    kjsArr = np.zeros((nVertsInGraph,nModules))
    csArr = np.zeros((nVertsInGraph,nModules))

    if centType=='node_strength':
        centVec = degreeOfROIs.copy()
    elif centType=='betweenness': # MATLAB original code uses 2 helper functions: weight_conversion.m (with 'lengths') and betweenness_wei.m 
        thisFC_Inverted = 1/fcArray # invert weights (element-wise)
        # centVec = betweenness_weigh(thisFC_Inverted) # *** TBA: adapt betweenness_wei as an extra helper function

    maxSummedCentrality = 0 
    for netNum in range(1,numNets+1): # +1 adjustment for py
        thisNetIx = np.where(thisNetAffilVec==netNum)[0]
        if np.nansum(centVec[thisNetIx]) > maxSummedCentrality:
            maxSummedCentrality = np.nansum(centVec[thisNetIx])

        netAffilMask_Binary = netAffilMask==netNum
        ksArr[:,netNum-1] = np.nansum(fcArray * netAffilMask_Binary,axis=1)

    for netNum in range(1,numNets+1):
        numNodesInNet = np.where(thisNetAffilVec==netNum)[0].shape[0]
        if numNodesInNet>1:
            thisNetIx = np.where(thisNetAffilVec==netNum)[0]
            ksHere = ksArr[thisNetIx,:].copy()
            ksSum = np.nansum(ksHere,axis=0)
            ksSum_Tile = np.tile(ksSum,(numNodesInNet,1)) # this is to replicate the matlab matrix multiplication effect (which in their use was just repeating the vector over rows)
            #kjsArr[thisNetIx,:] = np.matmul(np.ones(numNodesInNet),np.nansum(ksHere,axis=0))
            kjsArr[thisNetIx,:] = ksSum_Tile.copy()
            kjsArr[thisNetIx,netNum-1] = kjsArr[thisNetIx,netNum-1]/2 # account for redundancy

    for nodeNum in range(nVertsInGraph):
        if degreeOfROIs[nodeNum]>0:
            for netNum in range(1,numNets+1):
                fcNeighbors = fcArray[:,nodeNum].copy()
                fcNeighbors_Binary = fcNeighbors > 0
                netAffilsAdj = thisNetAffilVec * fcNeighbors_Binary # zero out network numbers corresponding to 0 FC weights (of neighbors of ROI)
                netIxsHereAdj = np.where(netAffilsAdj==netNum)[0]
                centHere = centVec[netIxsHereAdj].copy()
                csArr[nodeNum,netNum-1] = np.nansum(centHere)

    # Normalize
    ksNormed = ksArr/kjsArr # normalize total weight of connections per node by total connections
    nanIxsHereR,nanIxsHereC = np.where(np.isnan(ksNormed)) # account for NaNs from dividing by 0s in some cases
    ksNormed[nanIxsHereR,nanIxsHereC] = 0
    csNormed = csArr / maxSummedCentrality # normalize sum of centralities of neighbors by the max summed centrality

    # Compute gateway
    gsTotal = np.square((1-(ksNormed * csNormed))) # total weightings
    ksArr_Squared = np.square(ksArr) 
    degreeOfROIs_Squared = np.square(degreeOfROIs)
    gatewayCoef = 1-np.nansum(((ksArr_Squared.T/degreeOfROIs_Squared).T * gsTotal),axis=1) # use transpose to mimic matlab's ability to do element-wise 2D x 1D 

    # make sure NaNs and 0s = 0
    nanIxsGC = np.where(np.isnan(gatewayCoef))[0]
    gatewayCoef[nanIxsGC] = 0
    zeroIxsGC = np.where(gatewayCoef==0)[0]
    gatewayCoef[zeroIxsGC] = 0
    
    return gatewayCoef