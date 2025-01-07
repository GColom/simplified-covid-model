"""
This code has two aim:
1) Study sociality vs mobility, considering Bologna's traffic data of 2020, 2021, 2022.
2) Study the error measurements coming from the 7-days moving average, removing those data and performing an interpolation to fill the missing elements in the mobility array.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# Social activity rate
m_vals = np.array([1.00, 1.00, 1.00, 0.24, 0.16, 0.17, 0.18, 0.24, 0.24, 0.24, 0.34, 0.24, 0.21, 0.27, 0.25, 0.23, 0.29, 0.26, 0.20, 0.39, 0.29, 0.65, 0.49, 0.35, 0.29, 0.24, 0.19, 0.25, 0.06, 0.08, 0.18, 0.21, 0.18, 0.12, 0.08, 0.06])
m_days = np.array([0, 54, 61, 71, 83, 125, 139, 155, 167, 181, 258, 299, 320, 341, 359, 375, 398, 418, 429, 468, 482, 537, 560, 575, 610, 635, 688, 714, 727, 750, 770, 791, 810, 859, 900, 910])

url_2020 = './data/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
url_2021 = './data/rilevazione-autoveicoli-tramite-spire-anno-2021_header_mod.csv'
url_2022 = './data/rilevazione-autoveicoli-tramite-spire-anno-2022_header_mod.csv'

def get_total_mobility(url):

    '''
    Given the link of the dataframe, return an array containing the mobility.

            Parameters:
                    url (str): github url of the dataframe we want to save, i.e. 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
            Return:
                    mobility (np.array): array containing the number of motor veichles detected in that year, WITHOUT normalization or sample rolling.

    '''

    if '2020' in url:
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(url, sep = ';')


    col_list= ['0000 0100', '0100 0200', '0200 0300', "0300 0400" , "0400 0500" , "0500 0600" , "0600 0700" , "0700 0800" , "0800 0900" , "0900 1000" , "1000 1100" , "1100 1200" , "1200 1300" , "1300 1400" , "1400 1500" , "1500 1600" , "1600 1700" , "1700 1800" , "1800 1900" , "1900 2000" , "2000 2100" , "2100 2200" , "2200 2300" , "2300 2400"] 

    ordered_df = df.sort_values(by = 'data')
    ordered_df['Average Daily Mobility'] = ordered_df[col_list].sum(axis=1) / 24

    global_df = pd.pivot_table(
    ordered_df,
    index='data',
    columns='Nome via',
    values='Average Daily Mobility',
    aggfunc=np.sum,
    fill_value=0,
    margins=True,
    )

    total_mobility = global_df['All']
    lastElementIndex = len(total_mobility)-1
    total_mobility = total_mobility[:lastElementIndex]

    return total_mobility

mobility2020 = get_total_mobility(url_2020)
mobility2021 = get_total_mobility(url_2021)
mobility2022 = get_total_mobility(url_2022)
days = np.arange(1, 912) #912 is given by 366 + 365 + 181 (since the 2022 df finishes at 30th June)

total_mobility = pd.concat([mobility2020,mobility2021,mobility2022])#, ignore_index=True)
total_mobility.index = pd.to_datetime(total_mobility.index)
total_mobility = pd.DataFrame(total_mobility).rename(columns = {'All':'mobility'})
total_mobility = total_mobility.reindex(pd.date_range('2020-01-01', '2022-06-30'))

#index_list = [19, 97, 147, 243, 454, 512, 527, 635, 663, 700, 747, 777, 876]

problematic_dates = ["2020-04-13",
                     "2020-05-01",
                     "2020-06-02",
                     "2020-09-06",
                     "2020-12-08",
                     "2021-04-05",
                     "2021-05-01",
                     "2021-06-02",
                     "2021-06-17",
                     "2021-10-04",
                     "2021-11-01",
                     "2021-12-08",
                     "2022-02-23",
                     "2022-06-02"]

for date in problematic_dates:
    total_mobility.loc[date].mobility = np.nan
#%%
total_mobility = total_mobility.interpolate()
#%%
total_mobility['m_smoothed'] = total_mobility.mobility.rolling(7, center=True).mean()
#ax = total_mobility[["mobility", "m_smoothed"]].plot()
#%%
# Rolling mean outputs NaN values as first and last element of the array. Let's clean this.
# smoothed_total = smoothed_total.to_numpy()
# index_of_nan = np.argwhere(np.isnan(smoothed_total))

# smoothed_total = np.delete(smoothed_total,index_of_nan) #removing NaN values from the array
# days = np.delete(days, index_of_nan) # deleting days associated with NaN values

total_mobility['m_norm'] = total_mobility.m_smoothed/total_mobility[total_mobility.index < pd.to_datetime('2020-03-01')].m_smoothed.max() #normalization
# NdC: Normalisation is now performed with respect to pre-pandemic values.
#%%
sociality = pd.DataFrame(m_vals, index = m_days, columns = ['sociality'])

sociality75pc = pd.read_csv('data/sociality_variations.csv', index_col = 0)
sociality75pc.rename(columns = {"m_value_75pc" : "sociality75pc"}, inplace = True)

from datetime import datetime, timedelta

def days_to_datetime(days): 
    base_date = datetime(2020, 1, 1) 
    delta = timedelta(days=days) # Subtract 1 day since we start counting from January 1, 2020 
    result_date = base_date + delta 
    return result_date

tmp_sociality = pd.merge(left = sociality, right = sociality75pc, 
                         left_index = True, 
                         right_index = True)

tmp_sociality.index = [days_to_datetime(idx) for idx in tmp_sociality.index]

#%%

merged = pd.merge(left = total_mobility, right = tmp_sociality, how = 'left', left_index = True, right_index = True)

# Define breakpoints

s1lbp        = 7
s1rbp        = 20
s2lbp        = 21
s2rbp        = 24
s3lbp        = 25
s3rbp        = -1


# Define shift intervals
shift1_begin = days_to_datetime(int(m_days[s1lbp]))
shift1_ref   = days_to_datetime(int(m_days[s1lbp+1]))
shift1_end   = days_to_datetime(int(m_days[s1rbp]))
shift2_begin = days_to_datetime(int(m_days[s2lbp]))
shift2_ref   = days_to_datetime(int(m_days[s2lbp]))
shift2_end   = days_to_datetime(int(m_days[s2rbp]))
shift3_begin = days_to_datetime(int(m_days[s3lbp]))
shift3_ref   = days_to_datetime(int(m_days[s3lbp]))
shift3_end   = days_to_datetime(int(m_days[s3rbp]))

shift1     = merged.m_norm.loc[shift1_ref] - merged.sociality[shift1_ref]
shift175pc = merged.m_norm.loc[shift1_ref] - merged.sociality75pc[shift1_ref]

shift2     = merged.m_norm.loc[shift2_ref] - merged.sociality[shift2_ref]
shift275pc = merged.m_norm.loc[shift2_ref] - merged.sociality75pc[shift2_ref]

shift3     = merged.m_norm.loc[shift3_ref] - merged.sociality[shift3_ref]
shift375pc = merged.m_norm.loc[shift3_ref] - merged.sociality75pc[shift3_ref]

#%% Define shifted sociality and perform shift

merged["shifted_sociality"] = merged.sociality
merged["shifted_sociality75pc"] = merged.sociality75pc

merged.shifted_sociality[(merged.index >= shift1_begin) & (merged.index <= shift1_end)] += shift1
merged.shifted_sociality75pc[(merged.index >= shift1_begin) & (merged.index <= shift1_end)] += shift175pc

merged.shifted_sociality[(merged.index >= shift2_begin) & (merged.index <= shift2_end)] += shift2
merged.shifted_sociality75pc[(merged.index >= shift2_begin) & (merged.index <= shift2_end)] += shift275pc

merged.shifted_sociality[(merged.index >= shift3_begin) & (merged.index <= shift3_end)] += shift3
merged.shifted_sociality75pc[(merged.index >= shift3_begin) & (merged.index <= shift3_end)] += shift375pc

merged.to_csv("COVID_merged.csv")

#%%
# #Shifts
# shift_1 = total_mobility[m_days[8]] - m_vals[8] #165 -> 14 giugno

# shift_m_vals = np.copy(m_vals)

# for index in range(7,21): # dall'ottavo elemento in poi shifto tutti i valori di 0,57
#         shift_m_vals[index] += shift_1

# plt.scatter(days,total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
# plt.scatter(m_days, m_vals, label = 'Social Activity', color ='deeppink', marker = "+", s = 60) # normalized to 1
# plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
# plt.axvspan(153, 490, facecolor='lightsalmon', alpha=0.2,label = 'Shift = +' + str(round(shift_1,2)))

# m_vals = np.array([1.00, 1.00, 1.00, 0.24, 0.16, 0.17, 0.18, 0.24, 0.24, 0.24, 0.34, 0.24, 0.21, 0.27, 0.25, 0.23, 0.29, 0.26, 0.20, 0.39, 0.29, 0.65, 0.49, 0.35, 0.29, 0.24, 0.19, 0.25, 0.06, 0.08, 0.18, 0.21, 0.18, 0.12, 0.08, 0.06])
# 
# m_days = np.array([0, 54, 61, 71, 83, 125, 139, 155, 167, 181, 258, 299, 320, 341, 359, 375, 398, 418, 429, 468, 482, 537, 560, 575, 610, 635, 688, 714, 727, 750, 770, 791, 810, 859, 900, 910])

# plt.legend(loc="lower right")
# plt.title('Mobility & Sociality (shifted) vs Time')
# plt.show()

# shift_2 = total_mobility[m_days[25]] - shift_m_vals[25]
# for index in range(24,len(m_vals)):
#     shift_m_vals[index] += shift_2

# plt.scatter(days,total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
# plt.scatter(m_days, m_vals, label = 'Social Activity', color ='deeppink', marker = "+", s = 60) # normalized to 1
# plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
# plt.axvspan(153, 490, facecolor='lightsalmon', alpha=0.2,label = 'Shift = +' + str(round(shift_1,2)))
# plt.axvspan(602, 912, facecolor='yellow', alpha=0.2,label = 'Shift = +' + str(round(shift_2,2)))
# plt.legend(loc="lower right")
# plt.title('Mobility & Sociality (shifted) vs Time')
# plt.show()

# ---------------------------------------------------------------------
# Let's study the anomalous data given by the 7-days moving average (we may see sometimes 7 consecutive data shifted), in order to make our data smoother.

# def get_wrong_measurements(index_list):
    
#     '''
#     Function that marks the shifted measurements.

#             Parameters:
#                     index_list (list): list containing the first index of the total_mobility array in which wrong measurement start.
#             Return:
#                     errors_days (array): array containing the days related with the wrong mesurement.
#                     errors_total_mobility (array) : array containing the values of the wrong measurements.

#     '''

#     window = []
#     for element in index_list:
#         window += range(element, element + 7)
#     return days[window], total_mobility[window]

# index_list = [19, 97, 147, 243, 454, 512, 527, 635, 663, 700, 747, 777, 876]
# errors_days, errors_total_mobility = get_wrong_measurements(index_list)

# # plt.scatter(days,total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
# # plt.scatter(errors_days,errors_total_mobility, label = "Mobility wrong measurements", color = "red", marker = "o", s = 4)
# # plt.legend(loc="lower right")
# # plt.title('Mobility vs Time')
# # plt.show()

# # remove the shifted data
# index_to_remove = []
# for element in index_list:
#     index_to_remove += range(element, element + 7)

# total_mobility = np.delete(total_mobility, index_to_remove)
# days = np.delete(days, index_to_remove)
    
# #------------------------------------------
# # Interpolation of the (now) missing data.

# #operate a shift: necessary to correctly fill the missing point.
# index_to_remove = [element+4 for element in index_to_remove]

# def interpolation(x, y, kind):
    
#     '''
#     Given x, y, and type of the interpolation, perform the scatter interpolation in specific areas of the plot, and return x and y array with both orginal data and interpolation.

#             Parameters:
#                     x (np.array): 1-D array containing x values (here, days)
#                     y (np.array): 1-D array containing y values (here, mobility)
#                     kind (str): string that specifies the kind of interpolation, like "linear", "cubic", "quadratic" etc.
#             Return:
#                     x (np.array): 1-D array containing x original data + interpolation.
#                     y (np.array) : 1-D array containing y original data + interpolation.

#     '''

#     function = interpolate.interp1d(x, y, kind)

#     new_x = index_to_remove
#     interpolated_y = function(new_x)

#     # plt.scatter(days,total_mobility, label = "Original", color = "green", marker = "o", s = 4)
#     # plt.scatter(new_x, interpolated_y, label = str(kind) + " interpolation", color = 'red', marker = "o", s = 4)
#     # plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
#     # plt.axvline(x=731, color='darkturquoise', ls='dotted', label = 'End of the year')
#     # plt.legend(loc="lower right")
#     # plt.xlabel('Number of days from 1 January 2020')
#     # plt.ylabel('Average number of motor vehicle detected normalized')
#     # plt.title(str(kind) + " Interpolation in the missing point")
#     # plt.show()
    
#     y = np.append(y, interpolated_y)
#     x = np.append(x, new_x)

#     return x, y




# days, total_mobility = interpolation(days, total_mobility, kind = "linear")
#%% GRAPHICS

from matplotlib import dates

biyearly_loc = dates.MonthLocator( bymonth = [1, 7])

#Plots
fig1, ax1 = plt.subplots(figsize = (4.792, 4.792/np.sqrt(2)))
ax1.scatter(merged.index, merged.m_norm, color = "green", marker = "o", s = 4)
ax1.scatter(merged.index, merged.sociality, color ='red', marker = "+", s = 60)
ax1.scatter(merged.index, merged.sociality75pc, color ='blue', marker = "+", s = 60)

#plt.axvline(x=366, color='blue', ls='dotted')#, label = 'End of the year')
ax1.axvspan(days_to_datetime(70), days_to_datetime(125), facecolor='red', hatch = r'\\', edgecolor = 'red', alpha=0.2)#,
#plt.axvline(x=731, color='blue', ls='dotted')
ax1.axvline(x=days_to_datetime(297), color='orange', ls='--')#, label='Closure of activities') # 24th october: chiusura attività
ax1.axvline(x=days_to_datetime(319), color='red', ls='--')#, label = 'Night curfew and colour-coded zones') # 3th november, curfew, introduzione dei colori
# plt.title('2020-2022 Mobility vs Time')
ax1.set_xlabel(r'Date', fontsize = 12)

ax1.xaxis.set_major_locator(biyearly_loc)

# plt.ylabel('Average number of motor vehicle detected normalized')
#plt.legend(loc="lower right")
# plt.xlim([0, max(days)+1])
ax1.set_ylim(bottom = 0.)
fig1.tight_layout()
fig1.savefig("sociality_vs_mobility.pdf", dpi = 300)
plt.show()
#%%

fig2, ax2 = plt.subplots(figsize = (4.792, 4.792/np.sqrt(2)))
ax2.scatter(merged.index, merged.m_norm,  color = "green", marker = "o", s = 4)
#plt.axvline(x=366, color='blue', ls='dotted')#, label = 'End of the year')
#plt.axvline(x=731, color='blue', ls='dotted')
ax2.set_xlabel(r'Date', fontsize = 12)

ax2.xaxis.set_major_locator(biyearly_loc)


#plt.axvline(x=70, color='b', ls='--', label='1st Lockdown') # start of the first italian lockdown, 9th march
#plt.axvline(x=125, color='b', ls='--') # end lockdown, 18th may
ax2.axvspan(days_to_datetime(70), days_to_datetime(125), facecolor='red', hatch = r'\\', edgecolor = 'red', alpha=0.2)#,
            #label = '$1^{\mathrm{st}}$ national lockdown')
#
#plt.axvline(x=286, color='y', ls='--')#, label='Start of the 2nd wave') # start of the second wave, 13th october, mandatory mask
#plt.axvline(x=291, color='c', ls='--') # 18th october: chiusura scuola e università
ax2.axvline(x=days_to_datetime(297), color='orange', ls='--')#, label='Closure of activities') # 24th october: chiusura attività
ax2.axvline(x=days_to_datetime(319), color='red', ls='--')#, label = 'Night curfew and colour-coded zones') # 3th november, curfew, introduzione dei colori
# plt.xlim([0, max(days)+1])
ax2.set_ylim(bottom = 0.)
#plt.legend(loc="lower right")
fig2.tight_layout()
fig2.savefig("normalised_mobility.pdf", dpi = 300)
plt.show()

#%%
fig3, ax3 = plt.subplots(figsize = (4.792, 4.792/np.sqrt(2)))
ax3.scatter(merged.index, merged.m_norm,  color = "green", marker = "o", s = 4)
ax3.scatter(merged.index, merged.shifted_sociality,  color ='red', marker = "+", s = 60) # normalized to 1
ax3.scatter(merged.index, merged.shifted_sociality75pc,  color ='blue', marker = "+", s = 60) # normalized to 1

#plt.axvline(x=366, color='blue', ls='dotted')#, label = 'End of the year')
#plt.axvline(x=731, color='blue', ls='dotted')
ax3.axvspan(shift1_begin, shift1_end, facecolor='lightsalmon', hatch = r'\\', edgecolor = 'red', alpha=0.2,label = 'Shift = +' + str(round(shift1,2)))
#plt.axvspan(602, 912, facecolor='yellow', alpha=0.2,label = 'Shift = +' + str(round(shift_2,2)))
ax3.axvspan(shift2_begin, shift2_end, edgecolor='blue', hatch = r"//", alpha=0.2,label = 'Shift = +' + str(round(shift2,2)))
ax3.axvspan(shift3_begin, shift3_end, facecolor='green', edgecolor='darkgreen', hatch = r"x", alpha=0.15,label = 'Shift = +' + str(round(shift3,2)))
ax3.legend(loc="lower right")

ax3.xaxis.set_major_locator(biyearly_loc)

ax3.set_xlabel(r'Date', fontsize = 12)
# plt.title('Mobility & Sociality (shifted) vs Time')
# plt.xlim([0-1, max(days)+1])
ax3.set_ylim(bottom = 0.)
# plt.show()
fig3.tight_layout()
fig3.savefig("sociality_vs_mobility_shifted.pdf", dpi = 300)
plt.show()
#plt.close()
# plt.show()
#interpolation(days, total_mobility, kind = "quadratic")
#interpolation(days, total_mobility, kind = "cubic")

