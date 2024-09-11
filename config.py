#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:25:48 2024

@author: ainishlodaya
"""

import os

# Constants for generalization
TIME_PERIOD_SHORT = 20
TIME_PERIOD_LONG = 50
UPPER_ENVELOPE = 0.03
LOWER_ENVELOPE = 0.03
TIME_PERIOD_ATR = 14
WEEKLY_ROLLING_WINDOW = 20
VOLUME_MA_PERIOD = 20

# Default RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ATR multiplier for volatility breakout strategy
ATR_MULTIPLIER_DEFAULT = 1.5

# Plotting settings
PLOT_STYLE = 'ggplot'
PLOT_PARAMS = {
    'figure.figsize': (14, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.75,
    'axes.grid': True
}

# File paths and naming conventions
OUTPUT_DIR = os.getcwd() + '/'  # Set to current working directory
CSV_SUFFIX = '.csv'
PNG_SUFFIX = '.png'

# Debug settings
DEBUG_MODE = True
LOG_LEVEL = 'INFO'