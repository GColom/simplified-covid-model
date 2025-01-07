#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:33:33 2022

@author: Giulio Colombini
"""

import model as m
from getmobility import get_mobility

m_data = get_mobility('https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv')

translated_days = m_data[0] - 30

p_mask = translated_days > 0

t_data = (translated_days[p_mask], m_data[1][p_mask])

e = m.run_simulation(days = 100, m=t_data)

import matplotlib.pyplot as plt

for c in e[1:]:
    plt.plot(e[0], c)
    
plt.grid(True)
plt.xlim(0,100)
plt.ylim(0, 1000)