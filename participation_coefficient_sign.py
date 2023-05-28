# C. Cocuzza, 2022. Compute participation coefficicent (PC): converting brain connectivity toolbox code (MATLAB) to python.
# See here: https://drive.google.com/drive/folders/1eTYY4NAxMo7L2SLSyAUk8bI1JUHTIx7l
# See Guimera et al 2005; also Power's work on why PC > degree centrality for hubs.
# For connectivity arrays that are weighted and include positive/negative (signed) values.

# PC = distribution of a node's edges amongst the various (out-of-network or between-network) communities in a graph. 
# If PC = 0 for a given node, that node's edges are completely restricted to it's own community (within-network).  

# NOTE: a helper variable is included in this package called thisNetAffilVec.npy that can be used for the input thisNetAffilVec. This is based on the Cole Anticevic brain wide network partition (CABNP) (see Ji et al., 2019 or https://github.com/ColeLab/ColeAnticevicNetPartition). In addition, network names (in proper order 1-12, or 0-11 in python indexing) can be found in networkNamesCABNP_Long.npy (and corresponding acronyms in netNamesCABNP_Short.npy).

import numpy as np

def participation_coefficient_sign(fcArray,thisNetAffilVec,numNets):
    '''
    # INPUTS:
    # fcArray = a connectivity array of size: nodes x nodes x task conditions; either as-is or adjusted for pos/neg before hand (i.e., can mask for positive values, then run the function for just positive weights to get a positive PC); also sorted to network partition beforehand. NOTE: the recommended use is to pre-index fcArray over subjects and iteratively call this module, for example using fcArray[:,:,:,subject_index] as the fcArray input.
    # thisNetAffilVec = m, network assignment (i.e., affiliation) vector (note: start numbering at 1); this sorting should match fcArray
    # numNets = number of networks in thisNetAffilVec; should equal largest integer in that thisNetAffilVec

    # OUTPUTS: 
    # partCoef = m, vector of participation scores for each parent node 
    '''
    
    # Set up community affiliation on-diag array with binary mask; 0-weighted edges will be masked out
    # Note: b/c of python indexing, use +1 for network assignments
    nVertsInGraph = len(fcArray) # number of vertices
    degreeOfROIs = np.nansum(fcArray,axis=1) # degree/strength of "parent" regions (or targets in Cole lab convention)
    netAffilMask = np.matmul((fcArray!=0),(np.diag(thisNetAffilVec))) # community specific neighbors; note that * used in matlab = matrix multiplication
    #netAffilMask = np.dot((fcArray!=0),(np.diag(thisNetAffilVec))) # in this case matmul and dot would produce equal arrays
    
    # Build up degrees/strengths
    arrHere = np.zeros((nVertsInGraph));
    for netNum in range(1,numNets+1):
        # Note that element-wise multiplication (Hadamard) in PYTHON uses just the asterisk; but np.multiply() can be used too
        vecHere = fcArray * (netAffilMask==netNum) # equal to using np.muliply(fcArray,(netAffilMask==netNum))
        sumVec = np.nansum(vecHere,axis=1) # collapse 0s columns; NOTE: may need to deal with "-0"s here
        squareVec = np.square(sumVec) # can also use np.power(sumVec,2) or the double asterisk sumVec**2
        arrHere = arrHere + squareVec 

    partCoef = np.ones((nVertsInGraph))
    squareDegrees = np.square(degreeOfROIs)
    # to avoid divide by 0 errors, set to nan (/0 will = nan anyway, this just removes the error msg; nans handled later)
    zeroIxsSquareDegrees = np.where(squareDegrees==0)[0] 
    if zeroIxsSquareDegrees.shape[0]!=0:
        squareDegrees[zeroIxsSquareDegrees] = np.nan
    partCoef = partCoef - (arrHere/squareDegrees)
    
    # Find nodes with NaN participation coefficient and set PC score to 0
    nanInP = np.where(np.isnan(partCoef))[0]
    if not nanInP.shape[0]==0:
        partCoef[nanInP] = 0
    
    # Find nodes that had no out neighbors and set PC score to 0 (from original)
    zerosInFC = np.where(degreeOfROIs==0)[0] 
    if not zerosInFC.shape[0]==0:
        partCoef[zerosInFC] = 0 
        
    # Find nodes that had no out neighbors and set PC score to 0 (from _sign function)
    zerosInP = np.where(partCoef==0)[0] 
    if not zerosInP.shape[0]==0:
        partCoef[zerosInP] = 0 
        
    return partCoef