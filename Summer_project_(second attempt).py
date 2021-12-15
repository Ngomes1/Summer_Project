#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[5]:


raw_cvfs = pd.read_csv('Nepal' , sep='\t')
raw_cvfs.head()


# # Climate Data Information

# In[6]:


climate_data = pd.DataFrame(columns=['Year', 'Temp', 'Precip'])


# In[7]:


temp_annual = np.load('temp_annual.npy')
temp_annual2 = np.load('temp_annual2.npy')
Precip_annual = np.load('Precip_annual.npy')
Precip_annual2= np.load('Precip_annual2.npy')
climate_data['Year'] = np.arange(2001-1943,2021-1943, 1)
for i in range(10):
    climate_data['Temp'][i] = temp_annual[i]
    climate_data['Temp'][i + 10]= temp_annual2[i]
    climate_data['Precip'][i]= Precip_annual[i]
    climate_data['Precip'][i + 10] = Precip_annual2[i]

climate_data


# # Maize Data

# In[8]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

raw_cvfs = pd.read_csv('Nepal' , sep='\t')

Maize = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Maize['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'B12B' in item:
        Maize[item] = raw_cvfs[item].astype(float)
    elif 'B12A'in item:
        Maize[item] = raw_cvfs[item].astype(float)
    elif 'MIG' in item:
        Maize[item] = raw_cvfs[item].astype(float)
    elif 'REM' in item:
        Maize[item] = raw_cvfs[item].astype(float)
        
for col in Maize.columns:
    if ('MIG' in col) or ('REM' in col):
        Maize.rename(columns = {str(col):str(col)[0:3] + '_' + str(col)[-2:]}, inplace=True)
        

print(Maize.head())
Maize['B12B_64'].describe()
Maize_long = pd.wide_to_long(Maize, ['B12A', 'B12B', 'REM', 'MIG'], i = 'HHID', j = 'Year', sep = '_').reset_index()
print(Maize_long)


# In[9]:


Maize_climate = Maize_long.merge(climate_data, on = "Year", how = 'inner')
Maize_climate['temp^2'] = Maize_climate['Temp'] ** 2 
Maize_climate['precip^2'] = Maize_climate['Precip'] ** 2


# In[11]:


Maize_climate.tail()


# In[14]:


f,ax = plt.subplots()
graph = ax.scatter(Maize_long['Year'], Maize_long['B12B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Maize in Kg')
ax.set_xlabel('Year')
plt.show()
Maize_long.loc[Maize_long['Year']==64]['B12B'].mean()
#plt.savefig('Maize_Graph.png')


# In[13]:


Maize_drop = Maize_climate[Maize_climate.Year != 63] 
Maize_drop = Maize_drop[Maize_drop.Year != 72]
Maize_drop = Maize_drop[Maize_drop.B12B != 0]
print(Maize_drop)
graph = ax.scatter(Maize_drop['Year'], Maize_drop['B12B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Maize in Kg')
ax.set_xlabel('Year')
plt.show()
#plt.savefig('Maize_Graph.png')

f, ax1 = plt.subplots()

x = Maize_drop['REM']
y = Maize_drop['B12B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances and Climate')
ax1.set_ylabel('Maize Production (kg)')

plt.show()            


# In[22]:


#Corn = Maize_drop.loc[Maize_drop['Year'] == 64]
#Corn.mean()


# In[15]:


Maize_drop['REM'].describe()
             
Maize_drop['Yield'] = Maize_drop['B12B'] / Maize_drop['B12A']

y = Maize_drop['B12B']
        
x_1 = Maize_drop[['REM','Temp', 'Precip', 'temp^2', 'precip^2']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y.astype(float), x_1.astype(float))
est = est.fit()
est.summary()
#plt.savefig('Remittance to Maize Ratio')


# #              THE RICE SECTION  

# In[362]:


Rice = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Rice['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'B11B' in item:
        Rice[item] = raw_cvfs[item].astype(float)
        
    elif 'B11A' in item:
        Rice[item] = raw_cvfs[item].astype(float)
        
    elif 'MIG' in item:
        Rice[item] = raw_cvfs[item].astype(float)
        
    elif 'REM' in item:
        Rice[item] = raw_cvfs[item].astype(float)
        
for col in Rice.columns:
    if ('MIG' in col) or ('REM' in col):
        Rice.rename(columns = {str(col):str(col)[0:3] + '_' + str(col)[-2:]}, inplace=True)
        

print(Rice.head())
Rice['B11B_64'].describe()
Rice_long = pd.wide_to_long(Rice, ['B11A', 'B11B', 'REM', 'MIG'], i = 'HHID', j = 'Year', sep = '_').reset_index()
print(Rice_long)


# In[363]:


Rice_climate = Rice_long.merge(climate_data, on = "Year", how = 'inner')
Rice_climate['temp^2'] = Rice_climate['Temp'] ** 2 
Rice_climate['precip^2'] = Rice_climate['Precip'] ** 2


# In[364]:


Rice_climate.tail()


# In[249]:


f,ax = plt.subplots()
graph = ax.scatter(Rice_long['year'], Rice_long['B11B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Rice in Kg')
ax.set_xlabel('Year')
plt.show()
Rice_long.loc[Rice_long['year']==71]['B11B'].mean()
#plt.savefig('Rice_Graph.png')


# In[366]:


Rice_drop = Rice_climate[Rice_climate.Year != 63] 
Rice_drop = Rice_drop[Rice_drop.Year != 72]
print(Rice_drop)
graph = ax.scatter(Rice_drop['Year'], Rice_drop['B11B'])
ax.set_ylim([0, 2000])
ax.set_ylabel('Rice in Kg')
ax.set_xlabel('Year')
plt.show()
#plt.savefig('Rice_Graph.png')

f, ax1 = plt.subplots()

x = Rice_drop['REM']
y = Rice_drop['B11B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances')
ax1.set_ylabel('Rice Production (kg)')

plt.show()


# In[368]:


Rice_drop['REM'].describe()
             
Rice_drop['Yield'] = Rice_drop['B11B'] / Rice_drop['B11A']

y = Rice_drop['B11B']
        
x_1 = Rice_drop[['REM', 'Temp', 'Precip', 'temp^2', 'precip^2']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y.astype(float), x_1.astype(float))
est = est.fit()
est.summary()
#plt.savefig('Remittance to Rice Ratio')


# #                  WHEAT SECTION    

# In[252]:


Wheat = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Wheat['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'B13A' in item:
        Wheat[item] = raw_cvfs[item].astype(float)
    elif 'B13B' in item:
        Wheat[item] = raw_cvfs[item].astype(float)
    elif 'REM' in item:
        Wheat[item] = raw_cvfs[item].astype(float)
    elif 'MIG' in item:
        Wheat[item] = raw_cvfs[item].astype(float)
        
for col in Wheat.columns:
    if ('MIG' in col) or ('REM' in col):
        Wheat.rename(columns = {str(col):str(col)[0:3] + '_' + str(col)[-2:]}, inplace=True)
        

print(Wheat.head())
Wheat['B13B_64'].describe()
Wheat_long = pd.wide_to_long(Wheat, ['B13A', 'B13B', 'REM', 'MIG'], i = 'HHID', j = 'Year', sep = '_').reset_index()
print(Wheat_long)


# In[369]:


Wheat_climate = Wheat_long.merge(climate_data, on = "Year", how = 'inner')
Wheat_climate['temp^2'] = Wheat_climate['Temp'] ** 2 
Wheat_climate['precip^2'] = Wheat_climate['Precip'] ** 2


# In[370]:


Wheat_climate.tail()


# In[254]:


f,ax = plt.subplots()
graph = ax.scatter(Wheat_long['Year'], Wheat_long['B13B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Wheat in Kg')
ax.set_xlabel('Year')
plt.show()
Wheat_long.loc[Wheat_long['Year']==64]['B13B'].mean()
#plt.savefig('Wheat_Graph.png')


# In[371]:


Wheat_drop = Wheat_climate[Wheat_climate.Year != 63] 
Wheat_drop = Wheat_drop[Wheat_drop.Year != 72]
print(Wheat_drop)
graph = ax.scatter(Wheat_drop['Year'], Wheat_drop['B13B'])
ax.set_ylim([0, 2000])
ax.set_ylabel('Wheat in Kg')
ax.set_xlabel('Year')
plt.show()
#plt.savefig('Rice_Graph.png')

f, ax1 = plt.subplots()

x = Wheat_drop['REM']
y = Wheat_drop['B13B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances')
ax1.set_ylabel('Wheat Production (kg)')

plt.show()


# In[373]:


Wheat_drop['REM'].describe()
             
Wheat_drop['Yield'] = Wheat_drop['B13B'] / Wheat_drop['B13A']

y = Wheat_drop['B13B']
        
x_1 = Wheat_drop[['REM', 'Temp', 'Precip', 'temp^2', 'precip^2']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y.astype(float), x_1.astype(float))
est = est.fit()
est.summary()
#plt.savefig('Remittance to Wheat Ratio')


# #      MUSTARD SECTION   

# In[376]:


Mustard = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Mustard['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'B14B' in item:
        Mustard[item] = raw_cvfs[item].astype(float)    
    elif 'B14A' in item:
        Mustard[item] = raw_cvfs[item].astype(float)
    elif 'MIG' in item:
        Mustard[item] = raw_cvfs[item].astype(float)
    elif 'REM' in item:
        Mustard[item] = raw_cvfs[item].astype(float)
        
for col in Mustard.columns:
    if ('MIG' in col) or ('REM' in col):
        Mustard.rename(columns = {str(col):str(col)[0:3] + '_' + str(col)[-2:]}, inplace=True)
        

print(Mustard.head())
Mustard['B14B_64'].describe()
Mustard_long = pd.wide_to_long(Mustard, ['B14A', 'B14B', 'REM', 'MIG'], i = 'HHID', j = 'Year', sep = '_').reset_index()
print(Mustard_long)


# In[377]:


Mustard_climate = Mustard_long.merge(climate_data, on = "Year", how = 'inner')
Mustard_climate['temp^2'] = Mustard_climate['Temp'] ** 2 
Mustard_climate['precip^2'] = Mustard_climate['Precip'] ** 2


# In[378]:


Mustard_climate.head()


# In[397]:


f,ax = plt.subplots()
graph = ax.scatter(Mustard_long['Year'], Mustard_long['B14B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Mustard in Kg')
ax.set_xlabel('Year')
#plt.show()
#Mustard_long.loc[Mustard_long['year']==64]['B14B'].mean()
plt.savefig('Mustard_Graph')


# In[394]:


Mustard_drop = Mustard_climate[Mustard_climate.Year != 63] 
Mustard_drop = Mustard_drop[Mustard_drop.Year != 72]
print(Mustard_drop)
graph = ax.scatter(Mustard_drop['Year'], Mustard_drop['B14B'])
ax.set_ylim([0, 2000])
ax.set_ylabel('Mustard in Kg')
ax.set_xlabel('Year')
plt.show()
#plt.savefig('Mustard_Graph.png')

f, ax1 = plt.subplots()

x = Mustard_drop['REM']
y = Mustard_drop['B14B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances')
ax1.set_ylabel('Mustard Production (kg)')

#plt.show()
plt.savefig('Remittance to Mustard Ratio')


# In[382]:


Mustard_drop['REM'].describe()
             
Mustard_drop['Yield'] = Mustard_drop['B14B'] / Mustard_drop['B14A']

y = Mustard_drop['B14B']
        
x_1 = Mustard_drop[['REM', 'Temp', 'Precip', 'temp^2', 'precip^2']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y.astype(float), x_1.astype(float))
est = est.fit()
est.summary()
#plt.savefig('Remittance to Mustard Ratio')


# #  RED LENTIL SECTION   

# In[293]:


Red_lentil = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Red_lentil['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'B15B' in item:
        Red_lentil[item] = raw_cvfs[item].astype(float)
        
    elif 'B15A' in item:
        Red_lentil[item] = raw_cvfs[item].astype(float)
    elif 'REM' in item:
        Red_lentil[item] = raw_cvfs[item].astype(float)
    elif 'MIG'in item:
        Red_lentil[item] = raw_cvfs[item].astype(float)
        
for col in Red_lentil.columns:
    if ('MIG' in col) or ('REM' in col):
        Red_lentil.rename(columns = {str(col):str(col)[0:3] + '_' + str(col)[-2:]}, inplace=True)

print(Red_lentil.head())
Red_lentil['B15B_64'].describe()
Red_lentil_long = pd.wide_to_long(Red_lentil, ['B15B', 'B15C', 'REM', 'MIG'], i = 'HHID', j = 'Year', sep = '_').reset_index()
print(Red_lentil_long)


# In[384]:


Red_lentil_climate = Red_lentil_long.merge(climate_data, on = "Year", how = 'inner')
Red_lentil_climate['temp^2'] = Red_lentil_climate['Temp'] ** 2 
Red_lentil_climate['precip^2'] = Red_lentil_climate['Precip'] ** 2


# In[385]:


Red_lentil_climate.tail()


# In[398]:


f,ax = plt.subplots()
graph = ax.scatter(Red_lentil_long['Year'], Red_lentil_long['B15B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Red_lentil in Kg')
ax.set_xlabel('Year')
#plt.show()
#Red_lentil_long[Red_lentil_long['Year']==64]['B15B'].mean()
plt.savefig('Red_Lentil Graph')


# In[395]:


Red_lentil_drop = Red_lentil_climate[Red_lentil_climate.Year != 63] 
Red_lentil_drop = Red_lentil_drop[Red_lentil_drop.Year != 72]
print(Red_lentil_drop)
graph = ax.scatter(Red_lentil_drop['Year'], Red_lentil_drop['B15B'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('Red Lentil in Kg')
ax.set_xlabel('Year')
plt.show()
#plt.savefig('Red Lentil Graph.png')

f, ax1 = plt.subplots()

x = Red_lentil_drop['REM']
y = Red_lentil_drop['B15B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances')
ax1.set_ylabel('Red Lentil Production (kg)')

#plt.show()
plt.savefig('Remittance to Red_Lentil Ratio')


# In[387]:


Red_lentil_drop['REM'].describe()
             
Red_lentil_drop['Yield'] = Mustard_drop['B14B'] / Mustard_drop['B14A']

y = Red_lentil_drop['B15B']
        
x_1 = Red_lentil_drop[['REM', 'Temp', 'Precip', 'temp^2', 'precip^2']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y.astype(float), x_1.astype(float))
est = est.fit()
est.summary()
#plt.savefig('Remittance to Mustard Ratio'


# In[ ]:


###################        MIGRATION SECTION ################


# In[389]:


Migrants = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Migrants['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'MIG' in item:
        Migrants[item] = raw_cvfs[item].astype(float)
    elif 'REM' in item:
        Migrants[item] = raw_cvfs[item].astype(float)
        
        

print(Migrants.head())
Migrants['MIG64'].describe()
Migrants_long = pd.wide_to_long(Migrants, ['MIG', 'REM'], i = 'HHID', j = 'Year').reset_index()
print(Migrants_long)


# In[390]:


Migrants_drop = Migrants_long[Migrants_long.Year != 63] 
Migrants_drop = Migrants_drop[Migrants_drop.Year != 72]
print(Migrants_drop)
graph = ax.scatter(Migrants_drop['REM'], Migrants_drop['MIG'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('REM')
ax.set_xlabel('MIG')
plt.show()
#plt.savefig('Rice_Graph.png')

f, ax1 = plt.subplots()

x = Migrants_drop['MIG']
y = Migrants_drop['REM']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Migrants')
ax1.set_ylabel('Remitances in Rupees(10$^$7)')


plt.show()

Migrants_drop['REM'].describe()
             

y = Migrants_drop['REM']
        
x_1 = Migrants_drop[['MIG']]
    
x_1 = sm.add_constant(x_1)


est = sm.OLS(y, x_1)
est = est.fit()
est.summary()


# In[391]:


Migrants_long.head()


# In[393]:


f,ax = plt.subplots()
graph = ax.scatter(Migrants_long['Year'], Migrants_long['MIG'])
#ax.set_ylim([0, 2000])
ax.set_ylabel('# of Migrants')
ax.set_xlabel('Year')
#plt.show()
#plt.savefig('Migrantion_Graph.png')
Migrants_long.loc[Migrants_long['Year'] == 71]['MIG'].mean()


# In[116]:


Remitances = pd.DataFrame()
raw_cvfs.replace(' ', 0, inplace = True)
Remitances['HHID'] = raw_cvfs['HHID']
for item in raw_cvfs.columns:
    if 'REM' in item:
        Remitances[item] = raw_cvfs[item].astype(float)
        
   
        
    

print(Remitances.head())
Remitances['REM64'].describe()
Remitances_long = pd.wide_to_long(Remitances, ['REM'], i = 'HHID', j = 'year')
print(Remitances_long)


# In[117]:


Remitances_long['Year'] = 0
for index, row in Remitances_long.iterrows():
    Remitances_long['Year'][index] = index[1]


# In[118]:


Remitances_long.head()


# In[255]:


f,ax = plt.subplots()
graph = ax.scatter(Remitances_long['Year'], Remitances_long['REM'])
#ax.set_ylim([0, 10000000])
ax.set_ylabel('Remitances ( Rupees x $10^7$)')
ax.set_xlabel('Year')
plt.show()
#plt.savefig(‘Wheat_Graph.eps’) Save all immages using this syntax.


Remitances_long.loc[Remitances_long['Year'] == 64]['REM'].mean()


# In[ ]:


#################   FLOOD RATES  #############################################


# In[135]:


years = [raw_cvfs.A1_63, raw_cvfs.A1_64, raw_cvfs.A1_65, raw_cvfs.A1_66, raw_cvfs.A1_67, raw_cvfs.A1_68, raw_cvfs.A1_69, raw_cvfs.A1_70, raw_cvfs.A1_71, raw_cvfs.A1_72]
floods = np.zeros(len(years)) 
for index, item in enumerate(years):
    floods[index] = sum(item == 7)
print(floods)   


# In[ ]:


####################     RICE to REMITTANCE CORRELATIONS  ##################


# In[210]:


f, ax1 = plt.subplots()

x = Remitances_long['REM']
y = Rice_long['B11B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances (x $10^7$ Nepali Rupees)')
#ax1.set_xlim([0.0, 300000])
ax1.set_ylabel('Rice Production (kg)')

plt.show()


# In[ ]:


###########      MAIZE to REMITTANCE CORRELATIONS ###########


# In[149]:


f, ax1 = plt.subplots()

x = Remitances_long['REM']
y = Maize_long['B12B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances (x $10^7$ Nepali Rupees)')
ax1.set_xlim([0.0, 200000])
ax1.set_ylabel('Maize Production (kg)')

plt.show()


# In[148]:


f, ax1 = plt.subplots()

x = Remitances_long['REM']
y = Wheat_long['B13B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances (x $10^7$ Nepali Rupees)')
ax1.set_xlim([0.0, 400000])
ax1.set_ylabel('Wheat Production (kg)')

plt.show()


# In[ ]:


###########  Mustard to REMITTANCE CORRELATIONS ###########


# In[151]:


f, ax1 = plt.subplots()

x = Remitances_long['REM']
y = Mustard_long['B14B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances (x $10^7$ Nepali Rupees)')
ax1.set_xlim([0.0, 300000])
ax1.set_ylabel('Mustard Production (kg)')

plt.show()


# In[155]:


f, ax1 = plt.subplots()

x = Remitances_long['REM']
y = Red_lentil_long['B15B']
ax1.scatter(x, y)

m, b = np.polyfit(x, y, 1)
ax1.plot(x, m*x+b, color='red', ls='--')

print('y = %fx + %f'%(m, b))

ax1.set_xlabel('Remittances (x $10^7$ Nepali Rupees)')
#ax1.set_xlim([0.0, 500000])
ax1.set_ylabel('Red Lentil Production (kg)')

plt.show()


# %%
