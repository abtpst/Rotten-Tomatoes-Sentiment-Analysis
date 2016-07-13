'''
Created on Oct 2, 2015

@author: atomar
'''
def show_duplicates(df, cols=[], include_nulls=True):
    """
    # accepts a dataframe df and a column (or list of columns)
    # if list of columns is not provided - uses all df columns
    # returns a dataframe consisting of rows of df
    # which have duplicate values in "cols"
    # sorted by "cols" so that duplciates are next to each other
    # Note - doesn't change index values of rows
    """
    # ---------------------------------
    aa = df.copy()
    mycols = cols
    # ---------------------------------
    if len(mycols) <= 0:
        mycols = aa.columns.tolist()
    elif type(mycols) != list:
        mycols = list(mycols)
    # ---------------------------------
    if not include_nulls:
        mask = False
        for mycol in mycols:
            mask = mask | (aa[mycol] != aa[mycol])  # test for null values
        aa = aa[~mask]                              # remove rows with nulls in mycols
    if len(aa) <= 0:
        return aa[:0]
    # ---------------------------------
    # duplicated() method returns Boolean Series denoting duplicate rows
    mask = aa.duplicated(cols=mycols, take_last=False).values \
         | aa.duplicated(cols=mycols, take_last=True).values
    aa = aa[mask]
    if len(aa) <= 0:
        return aa[:0]
    # ---------------------------------
    # sorting to keep duplicates together
    # Attention - can not sort by nulls
    # bb contains mycols except for cols which are completely nulls
    bb = aa[mycols]
    bb = bb.dropna(how='all',axis=1)
    # sort aa by columns in bb (thus avoiding nulls)
    aa = aa.sort_index(by=bb.columns.tolist())
    # ---------------------------------
    # sorting skips nulls thus messing up the order. 
    # Let's put nulls at the end
    mask = False
    for mycol in mycols:
        mask = mask | (aa[mycol] != aa[mycol])  # test for null values
    aa1 = aa[~mask]
    aa2 = aa[mask]
    aa = aa1.append(aa2)

    return aa

import pandas as pd
def combineRes(senList,ind, classifierType):
    
  
    combineDframe = pd.DataFrame(columns=['PhraseId','Sentiment'])
    
   
    tot = 0
    
    for suff in senList:
        
        partialRes = pd.read_csv('../../submits/pipeLine/tags/Tagpipeline'+classifierType+suff+'.csv', header=0, delimiter=",", quoting=3 )
    
        partPos = partialRes.loc[partialRes.Sentiment == 1]
        partNeg = partialRes.loc[partialRes.Sentiment == 0]
        
        partPos['Sentiment'] = suff
        
        print("For "+str(suff)+" " + str(len(partPos[partPos.Sentiment == suff])))
        
        combineDframe = combineDframe.append(partPos, True)
       
        tot = tot + len(partPos[partPos.Sentiment == suff])
        
    counts = combineDframe['PhraseId'].value_counts()
    
    values = [x for x in counts.index if counts[x]>1]
    
    result = []
    for e in values:
        vals = combineDframe[combineDframe['PhraseId'] == e].Sentiment.value_counts().index.values
        result.append((e, vals))
    
    junkIdx = dupHandler(result,combineDframe)
    
    for j in junkIdx:
        
        combineDframe = combineDframe.drop(j)
        
    print(len(combineDframe.PhraseId.unique()))
    print(len(combineDframe.PhraseId))
    print(combineDframe[combineDframe.PhraseId==int(193786.0)])
    
    #pp = show_duplicates(combineDframe, ["PhraseId"],True)
    
    return combineDframe

def dupHandler(dupArray, resData):
    
    res = resData
    
    idx = []
    
    for x in dupArray:
        
        resSent = getSent(x[1])
      
        idx.append(res[(res.PhraseId == int(x[0])) & (res.Sentiment != resSent)].index.tolist())
              
    return idx
    
def getSent(sentArray):
   
    if "Neutral" in sentArray:
        return "Neutral"
    
    if "SomewhatNegative" in sentArray and "Negative" in sentArray:
        return"SomewhatNegative"
    
    if "SomewhatPositive" in sentArray and "Positive" in sentArray:
        return"SomewhatPositive"
    
    return "Neutral"