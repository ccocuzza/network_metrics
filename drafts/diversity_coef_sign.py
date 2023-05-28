# C. Cocuzza, 2022. Compute Shannon entropy/diversity coefficicent (DC): converting brain connectivity toolbox code (MATLAB) to python.
# Based on code here: https://drive.google.com/drive/folders/1eTYY4NAxMo7L2SLSyAUk8bI1JUHTIx7l
# See Shannon 1948; Rubinov and Sporns 2011

# This measures diversity of intermodular connections; per node. Ranges from 0 to 1.

# This code is for the general form (for weighted, undirected graphs); use separate pos/neg inputs accordingly 

import numpy as np 

def diversity_coef_sign(fcArray,thisNetAffilVec,numNets):
    '''
    # INPUTS:
    # fcArray = a connectivity array of size: nodes x nodes x task conditions; either as-is or adjusted for pos/neg before hand (i.e., can mask for positive values, then run the function for just positive weights to get a positive DC); also sorted to network partition beforehand. NOTE: the recommended use is to pre-index fcArray over subjects and iteratively call this module, for example using fcArray[:,:,:,subject_index] as the fcArray input.
    # thisNetAffilVec = m, network assignment (i.e., affiliation) vector (note: start numbering at 1); this sorting should match fcArray
    # numNets = number of networks in thisNetAffilVec; should equal largest integer in that thisNetAffilVec

    # OUTPUTS: 
    # shannonDiversity = m, vector of diversity coefficients for each parent node 
    '''
    nVertsInGraph = int(len(fcArray))
    nModules = int(np.nanmax(thisNetAffilVec))
    degreeOfROIs = np.nansum(fcArray,axis=1)
    
    # Node-to-module degree
    arrHere = np.zeros((nVertsInGraph,nModules))
    for netNum in range(1,nModules+1): # +1 bc py indexing (but need non-zero network numbers in function)
        netIxsHere = np.where(thisNetAffilVec==netNum)[0]
        arrHere[:,netNum-1] = np.nansum(fcArray[:,netIxsHere],axis=1)
        
    repDegrees = np.transpose([degreeOfROIs] * nModules) # repeat the sum vec from above into columns of an arr (num of columns = num of nets); note that brackets are needed around var
    pArrHere = arrHere / repDegrees # elementwise divide
    
    # Find nodes with NaN and set to 0 
    nanIxsHereR,nanIxsHereC = np.where(np.isnan(pArrHere))
    pArrHere[nanIxsHereR,nanIxsHereC] = 0
    
    # Find nodes with 0 scores and set to 1 
    zeroIxsHereR,zeroIxsHereC = np.where(pArrHere==0)
    pArrHere[zeroIxsHereR,zeroIxsHereC] = 1
    
    # Compute "H" or diversity coefficient
    pArrHere_Log = np.log(pArrHere)
    nModules_Log = np.log(nModules)
    # Note that element-wise multiplication (Hadamard) in PYTHON uses just the asterisk; but np.multiply() can be used too
    pArrHere_Transformed = pArrHere * pArrHere_Log
    shannonDiversity = (np.sum(pArrHere_Transformed,axis=1)/nModules_Log) * -1
        
    return shannonDiversity