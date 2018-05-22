#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:19:13 2018

@author: sebastianhiguita
"""

import numpy as np

a = np.array((1, 0, 0))
b = np.array((0, 1, 0))

print(a);
print(b)

dist = np.linalg.norm(a - b)
print(dist)