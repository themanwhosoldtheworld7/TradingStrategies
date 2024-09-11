#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:39:32 2024

@author: ainishlodaya
"""

# main.py

from trading_strategies2 import TradingStrategy

def main():
    # Strategy 1: Moving Average Crossover with RSI Confirmation (Google)
    google_strategy = TradingStrategy(ticker='TCS.NS')
    google_strategy.dynamic_fibonacci_retracement_with_trailing_stop()

    # Strategy 2: Volatility Breakout with 1.5 ATR (Apple)
    apple_strategy = TradingStrategy(ticker='TCS.NS')
    apple_strategy.volatility_breakout_strategy()

    # Strategy 3: Moving Average Envelope (Pfizer)
    pfizer_strategy = TradingStrategy(ticker='TCS.NS')
    pfizer_strategy.moving_average_envelope_strategy2()

    # Strategy 4: Multi-Time Frame Breakout (Coca-Cola)
    coca_cola_strategy = TradingStrategy(ticker='TCS.NS')
    coca_cola_strategy.multi_time_frame_breakout_strategy()

if __name__ == '__main__':
    main()