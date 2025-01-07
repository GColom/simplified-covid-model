#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:33:33 2022

@author: Giulio Colombini
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


full = pd.read_csv("./virus_mononodo_Bologna_2022_03_15_ridotto.csv", 
                    skipinitialspace = True,
                    usecols = ['tempo_dati', 'Var[CONTAGI-spe]', 'Var[CONTAGI-spe-lisc]',
                               'tempo_sim', 'Var[CONTAGI (previsione)]'])

full.rename({'tempo_dati' : 'data_time', 'Var[CONTAGI-spe]' : 'measured', 
             'Var[CONTAGI-spe-lisc]' : 'smoothed', 'tempo_sim' : 'sim_time', 
             'Var[CONTAGI (previsione)]' : 'prediction'}, inplace = True, axis = 1)

from datetime import datetime, timedelta

def days_to_datetime(days): 
    base_date = datetime(2020, 1, 1) 
    delta = timedelta(days=days) # Subtract 1 day since we start counting from January 1, 2020 
    result_date = base_date + delta 
    return result_date


from matplotlib import dates

biyearly_loc = dates.MonthLocator(interval = 4)

# FIRST YEAR

fig1, ax1 = plt.subplots(figsize = (3 + 3/8, (3 + 3/8)/(2**0.5))) #(4.792, 4.792/np.sqrt(2)))

data_year_one_mask= (~full.measured.isna())   & (full.data_time<= 366)
sim_year_one_mask = (~full.prediction.isna()) & (full.sim_time <= 366)


yearone_datetime_data = [days_to_datetime(day) for day in full[data_year_one_mask].data_time.to_numpy()]

ax1.scatter(yearone_datetime_data, 
            full.measured[data_year_one_mask].to_numpy(), 
            label = 'Daily new positive swabs', s = 5)
ax1.scatter(yearone_datetime_data, 
            full[data_year_one_mask].smoothed.to_numpy(), 
            label = '7-day running average', s = 5, marker = "v")

yearone_datetime_sim = [days_to_datetime(day) for day in full[sim_year_one_mask].sim_time.to_numpy()]

ax1.plot(yearone_datetime_sim, 
        full[sim_year_one_mask].prediction.to_numpy(), 
        label = 'Full model prediction', linestyle = "solid", color = "C3")

ax1.set_xlabel("Date", fontsize = 11)
ax1.set_ylabel("Number of individuals", fontsize = 11)
ax1.xaxis.set_major_locator(dates.MonthLocator(interval = 4))
ax1.set_xlim(days_to_datetime(0), days_to_datetime(366))
ax1.set_ylim(bottom = 0.)
#ax1.legend()
#ax1.grid(True)
fig1.tight_layout()
fig1.savefig("mononodo2020.pdf", dpi = 300)
plt.show()

# SECOND YEAR

fig2, ax2 = plt.subplots(figsize = (3 + 3/8, (3 + 3/8)/(2**0.5))) #(4.792, 4.792/np.sqrt(2)))

data_year_two_mask= (~full.measured.isna())   & (full.data_time > 366)& (full.data_time <= 731)
sim_year_two_mask = (~full.prediction.isna()) & (full.sim_time > 366) & (full.sim_time <= 731)

yeartwo_datetime_data = [days_to_datetime(day) for day in full[data_year_two_mask].data_time.to_numpy()]


ax2.scatter(yeartwo_datetime_data, 
            full.measured[data_year_two_mask].to_numpy(), 
            label = 'Daily new positive swabs', s = 5)
ax2.scatter(yeartwo_datetime_data, 
            full[data_year_two_mask].smoothed.to_numpy(), 
            label = '7-day running average', s = 5, marker = "v")

yeartwo_datetime_sim = [days_to_datetime(day) for day in full[sim_year_two_mask].sim_time.to_numpy()]

ax2.plot(yeartwo_datetime_sim, 
        full[sim_year_two_mask].prediction.to_numpy(), 
        label = 'Full model prediction', linestyle = "solid", color = "C3")

ax2.set_xlabel("Date", fontsize = 11)
ax2.set_ylabel("Number of individuals", fontsize = 11)
ax2.xaxis.set_major_locator(dates.MonthLocator(interval = 4))
ax2.set_xlim(days_to_datetime(366), days_to_datetime(731))
ax2.set_ylim(bottom = 0.)

#ax2.legend()
#ax2.grid(True)
fig2.tight_layout()
fig2.savefig("mononodo2021.pdf", dpi = 300)
plt.show()
