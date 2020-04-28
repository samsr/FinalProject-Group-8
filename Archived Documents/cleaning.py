import pandas as pd
import numpy as np

"""
Some of the cleaning and coding for modeling was done in Excel, because it was easier to automatically
convert things like times and dates to numerical variables there by filling.
"""

class cleaner():

    def __init__(self):
        self.importData()
        #self.data.apply(self.cleanCol, axis = 0)
        print(self.data.shape)
        self.dropRows()
        print(self.data.shape)
        #self.testing()
        self.cleanAges()
    def importData(self):
        self.data = pd.read_csv("df_2019-2020.csv", low_memory=False)#.iloc[:10000,]
        #print(self.data)
        print("Imported")

    def dropRows(self):
        #indexNames = self.data[self.data["VIC_AGE_GROUP"]]
        drops = []
        for i, row in self.data.iterrows():
            if (row["VIC_AGE_GROUP"] not in ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]): drops.append(i)
            if (row["BORO_NM"] == np.nan): drops.append(i)
            if (row["JURISDICTION_CODE"] == np.nan): drops.append(i)
            if (row["PREM_TYPE_DESC"] == np.nan): drops.append(i)
            if (i%50000 == 0): print (i)
        self.data = self.data.drop(self.data.index[list(set(drops))])


    def testing(self):
        print(self.data.iloc[1,4])

    def cleanCol(self, series):
        col = series.name
        if (col == "VIC_AGE_GROUP"): return series.apply(self.cleanAges)
        else: return series

    def cleanAges(self):
        for 
        if (e not in ["<18", "18-24", "25-44", "45-64", "65+"])

a = cleaner()

