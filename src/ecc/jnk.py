'''
Created on Oct 8, 2015

@author: atomar
'''

import pandas as pd
eccDframe = pd.DataFrame(columns=['PhraseId'])

x = pd.DataFrame(columns=['S'])
x["S"]=[1,2,3,4,5,6]

eccDframe.PhraseId = [12,13,14,15,16,17]
 
#eccDframe = pd.concat([x["S"]], axis = 1, keys=['Sentiment '])
eccDframe=eccDframe.join(x["S"])
eccDframe.rename(columns={'S':'X'},inplace=True)

x = pd.DataFrame(columns=['S'])
x["S"]=[1,2,3,4,5,6]

eccDframe=eccDframe.join(x["S"])
eccDframe.rename(columns={'S':'Z'},inplace=True)


print(eccDframe)
 
 