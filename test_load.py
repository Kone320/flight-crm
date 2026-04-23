#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:07:18 2026

@author: konearounaromeo
"""

from utils.data_utils import load_dataset
df = load_dataset(sample_size=500)
print(len(df))
print(df.columns)