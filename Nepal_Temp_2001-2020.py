#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
import numpy as np
import xarray as xr
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt


# In[7]:


fp = 'cru_ts4.05.2001.2010.tmp.dat.nc.gz'

temp_data = xr.open_dataset(fp)

nepal_temp = temp_data.loc[dict(lat=slice(26,28), lon=slice(80.5,88))]

print(nepal_temp)


# In[8]:


chitwan_temp = nepal_temp.loc[dict(lat = 27.75, lon = 84.25 )]

print(chitwan_temp)


# In[9]:


years = 10 

temp_annual = np.zeros(years)

for year in range(years):
    
    start_index = year * 12 + 3 
    end_index = (year + 1) * 12
    
    temp_annual[year] = chitwan_temp['tmp'][start_index:end_index].mean()
    
    
print(temp_annual)
np.save('temp_annual.npy', temp_annual)


# In[57]:


f, ax = plt.subplots()

x = np.arange(2001, 2011, 1)
y = temp_annual

ax.set_xlabel('Year')
ax.set_ylabel('Mean Annual Temp ($^0$C)')
ax.plot(x, temp_annual)

#plt.show()
plt.savefig('temperature')
print(temp_annual[4])


# In[16]:


fp2 = 'cru_ts4.05.2011.2020.tmp.dat.nc.gz'

temp_data2 = xr.open_dataset(fp2)

nepal_temp2 = temp_data2.loc[dict(lat=slice(26,28), lon=slice(80.5,88))]

print(nepal_temp2)


# In[17]:


chitwan_temp2 = nepal_temp2.loc[dict(lat = 27.75, lon = 84.25 )]

print(chitwan_temp2)


# In[52]:


years2 = 10 

temp_annual2 = np.zeros(years2)

for year in range(years2):
    
    start_index2 = year * 12
    end_index2 = (year + 1) * 12
    
    temp_annual2[year] = chitwan_temp2['tmp'][start_index2:end_index2].mean()
    
    
print(temp_annual2)
np.save('temp_annual2.npy', temp_annual2)


# In[19]:


f, ax = plt.subplots()

x = np.arange(2011, 2021, 1)
y = temp_annual2

ax.set_xlabel('Year')
ax.set_ylabel('Mean Annual Temp ($^0$C)')
ax.plot(x, temp_annual2)

plt.show()


# In[29]:


import sys
#!{sys.executable} -m pip install netCDF4
#!{sys.executable} -m pip install numpy --upgrade
#!{sys.executable} -m pip install xarray
get_ipython().system('{sys.executable} -m pip install scipy --upgrade')
import numpy as np
import xarray as xr
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt


# In[34]:


Precip = 'cru_ts4.05.2001.2010.pre.dat.nc.gz'

precipitation_data = xr.open_dataset(Precip)

nepal_rain = precipitation_data.loc[dict(lat=slice(26,28), lon=slice(80.5,88))]

print(nepal_rain)


# In[38]:


Chitwan_rain = nepal_rain.loc[dict(lat = 27.75, lon = 84.25 )]

Chitwan_rain


# In[53]:


years = 10 

Precip_annual = np.zeros(years)

for year in range(years):
    
    start_index2 = year * 12
    end_index2 = (year + 1) * 12
    
    Precip_annual[year] = Chitwan_rain['pre'][start_index2:end_index2].mean()
    
    
print(Precip_annual)
np.save('Precip_annual.npy', Precip_annual)


# In[47]:


f, ax = plt.subplots()

x = np.arange(2001, 2011, 1)
y = Precip_annual

ax.set_xlabel('Year')
ax.set_ylabel('Mean Annual Precipitation (in MM)')
ax.plot(x, Precip_annual)

#plt.show()
plt.savefig('Precipitation.png')


# In[41]:


Precip2 = 'cru_ts4.05.2011.2020.pre.dat.nc.gz'

precipitation_data2 = xr.open_dataset(Precip2)

nepal_rain2 = precipitation_data2.loc[dict(lat=slice(26,28), lon=slice(80.5,88))]

print(nepal_rain2)


# In[42]:


Chitwan_rain2 = nepal_rain2.loc[dict(lat = 27.75, lon = 84.25 )]

print(Chitwan_rain2)


# In[55]:


years = 10 

Precip_annual2 = np.zeros(years)

for year in range(years):
    
    start_index2 = year * 12
    end_index2 = (year + 1) * 12
    
    Precip_annual2[year] = Chitwan_rain2['pre'][start_index2:end_index2].mean()
    
    
print(Precip_annual2)
np.save('Precip_annual2.npy', Precip_annual2)


# In[45]:


f, ax = plt.subplots()

x = np.arange(2011, 2021, 1)
y = Precip_annual2

ax.set_xlabel('Year')
ax.set_ylabel('Mean Annual Precipitation (in MM)')
ax.plot(x, Precip_annual2)

plt.show()


# In[ ]:




