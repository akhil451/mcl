#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:26:20 2017

@author: akhil
"""

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

df=pd.read_csv("Restaurant_Reviews.tsv",sep="\t",quoting=3)
# qouting=3 -- for ignoring the qoutes .
# https://stackoverflow.com/questions/43344241/quoting-parameter-in-pandas-read-csv

import re
review = re.sub('[^a-zA-Z]'," ",df["Review"][0])
review=review.lower()
