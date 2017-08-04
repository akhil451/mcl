import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df=pd.read_csv("Ads_CTR_Optimisation.csv")

d=10  # no of ads 
N=10000 # no of users 
no_selected=[0]*d
sum_of_rewards=[0]*d
total_reward =0
ad_selected=[]

for n in range(N):
   max_upperbound =0
   ad=0
   for i in range(d):
        if(no_selected[i]>0):
            average_reward=sum_of_rewards[i]/no_selected[i]
            delta=np.sqrt((3/2)*np.log(n+1)/no_selected[i])
            upperbound=average_reward+delta
        else:
            upperbound = 1e400
        if(upperbound>max_upperbound):
         max_upperbound=upperbound
         ad=i
   ad_selected.append(ad)
   no_selected[ad] =  no_selected[ad] + 1
   reward = df.values[n, ad]
   sum_of_rewards[ad] = sum_of_rewards[ad] + reward
   total_reward = total_reward + reward
       
# visualising 
plt.hist(ad_selected) 

plt.show()