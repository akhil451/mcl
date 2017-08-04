import pandas as pd 
import seaborn as sns
import random
import matplotlib.pyplot as plt
sns.set()
df=pd.read_csv("Ads_CTR_Optimisation.csv")

d=10  # no of ads 
N=10000 # no of users 

total_reward =0
ad_selected=[]
no_of_times_1=[0]*d
no_of_times_0=[0]*d
for n in range(N):
    max_rand=0
    ad=0
    for i in range(d):
        rand_beta = random.betavariate(no_of_times_0[i]+1,no_of_times_1[i]+1)
        if(rand_beta>max_rand):
            max_rand=rand_beta
            ad=i
    ad_selected.append(ad)
    reward = df.values[n, ad]
    if(reward==1):
     no_of_times_0[ad]+=1
    else :
     no_of_times_1[ad]+=1  
    total_reward+=reward 
     
# visualising 
plt.hist(ad_selected) 
plt.ylabel('AD Click through frequency')
plt.xlabel('AD Number')
plt.show()