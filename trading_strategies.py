import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *  # Import everything from config.py

# Apply plotting settings
plt.style.use(PLOT_STYLE)
plt.rcParams.update(PLOT_PARAMS)

class TradingStrategy:
    def __init__(self, ticker, **kwargs):
        self.ticker = ticker
        self.kwargs = kwargs
        self.data = None

    def fetch_data(self, period='5y'):
        """Fetch historical data for the given ticker and period."""
        self.data = yf.download(self.ticker, period=period)
    
    def moving_average_crossover_with_rsi(self, period='5y', 
                                          time_period_short=TIME_PERIOD_SHORT, 
                                          time_period_long=TIME_PERIOD_LONG):
        """Strategy 1: Moving Average Crossover with RSI Confirmation."""
        self.fetch_data(period)
        short_ma_label = f'MA{time_period_short}'
        long_ma_label = f'MA{time_period_long}'
        self.data[short_ma_label] = ta.SMA(self.data['Close'], timeperiod=time_period_short)
        self.data[long_ma_label] = ta.SMA(self.data['Close'], timeperiod=time_period_long)
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        self.data['Signal'] = 0
        
        # Buy and Sell signals
        self.data.loc[(self.data[short_ma_label].shift(1) <= self.data[long_ma_label].shift(1)) & 
                      (self.data[short_ma_label] > self.data[long_ma_label]) & 
                      (self.data['RSI'] > 50), 'Signal'] = 1
        self.data.loc[(self.data[short_ma_label].shift(1) >= self.data[long_ma_label].shift(1)) & 
                      (self.data[short_ma_label] < self.data[long_ma_label]) & 
                      (self.data['RSI'] < 50), 'Signal'] = -1

        # Plot with consistent style
        plt.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.8)
        plt.plot(self.data.index, self.data[short_ma_label], label=f'{time_period_short}-Day MA', alpha=0.8)
        plt.plot(self.data.index, self.data[long_ma_label], label=f'{time_period_long}-Day MA', alpha=0.8)
        plt.plot(self.data[self.data['Signal'] == 1].index, self.data['Close'][self.data['Signal'] == 1], 
                 '^', markersize=10, color='green', lw=0, label='Buy Signal')
        plt.plot(self.data[self.data['Signal'] == -1].index, self.data['Close'][self.data['Signal'] == -1], 
                 'v', markersize=10, color='red', lw=0, label='Sell Signal')
        plt.title(f'{self.ticker} - Moving Average Crossover with RSI Confirmation')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        self.data.to_csv(f'{OUTPUT_DIR}CrossOverStrategy{CSV_SUFFIX}')

    def volatility_breakout_strategy(self, period='2y', 
                                     time_period_atr=TIME_PERIOD_ATR, 
                                     atr_multiplier=ATR_MULTIPLIER_DEFAULT):
        """Strategy 2: Volatility Breakout with ATR."""
        self.fetch_data(period)
        self.data = self.data[['Close', 'High', 'Low']].copy()
        self.data['ATR'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=time_period_atr)
        self.data['Upper_Breakout'] = self.data['Close'].shift(1) + atr_multiplier * self.data['ATR']
        self.data['Lower_Breakout'] = self.data['Close'].shift(1) - atr_multiplier * self.data['ATR']
        self.data['Buy_Signal'] = np.where(self.data['Close'] > self.data['Upper_Breakout'], 1, 0)
        self.data['Sell_Signal'] = np.where(self.data['Close'] < self.data['Lower_Breakout'], -1, 0)
        self.data['Signal'] = self.data['Buy_Signal'] + self.data['Sell_Signal']

        # Plot with consistent style
        plt.plot(self.data['Close'], label='Close Price', color='blue', alpha=0.8)
        plt.plot(self.data['Upper_Breakout'], label=f'Upper Breakout Level ({atr_multiplier} ATR)', color='green', linestyle='--')
        plt.plot(self.data['Lower_Breakout'], label=f'Lower Breakout Level ({atr_multiplier} ATR)', color='red', linestyle='--')
        plt.plot(self.data.loc[self.data['Buy_Signal'] == 1].index, self.data['Close'][self.data['Buy_Signal'] == 1], 
                 '^', markersize=10, color='green', lw=0, label='Buy Signal')
        plt.plot(self.data.loc[self.data['Sell_Signal'] == -1].index, self.data['Close'][self.data['Sell_Signal'] == -1], 
                 'v', markersize=10, color='red', lw=0, label='Sell Signal')
        plt.title('Volatility Breakout Strategy with ATR')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}VolatilityBreakoutStrategy_ATR{PNG_SUFFIX}', format='png')
        plt.show()
        self.data.to_csv(f'{OUTPUT_DIR}VolatilityBreakoutStrategy_ATR{CSV_SUFFIX}')

    def moving_average_envelope_strategy(self, period='2y', 
                                         upper_envelope=UPPER_ENVELOPE, 
                                         lower_envelope=LOWER_ENVELOPE, 
                                         time_period_short=TIME_PERIOD_SHORT):
        """Strategy 3: Moving Average Envelope."""
        self.fetch_data(period)
        ema_string = f'EMA{time_period_short}'
        self.data[ema_string] = ta.EMA(self.data['Close'], timeperiod=time_period_short)
        self.data['UpperEnvelope'] = self.data[ema_string] * (1 + upper_envelope)
        self.data['LowerEnvelope'] = self.data[ema_string] * (1 - lower_envelope)
        self.data['Signal'] = 0  
        self.data['Signal'][self.data['Close'] <= self.data['LowerEnvelope']] = 1
        self.data['Signal'][self.data['Close'] >= self.data['UpperEnvelope']] = -1

        # Plot with consistent style
        plt.plot(self.data['Close'], label='Close Price', color='blue', alpha=0.8)
        plt.plot(self.data[ema_string], label=f'{time_period_short}-Day EMA', color='orange')
        plt.plot(self.data['UpperEnvelope'], label=f'Upper Envelope ({upper_envelope*100:.1f}%)', color='red', linestyle='--')
        plt.plot(self.data['LowerEnvelope'], label=f'Lower Envelope ({lower_envelope*100:.1f}%)', color='green', linestyle='--')
        plt.plot(self.data[self.data['Signal'] == 1].index, self.data['Close'][self.data['Signal'] == 1], 
                 '^', markersize=10, color='green', lw=0, label='Buy Signal')
        plt.plot(self.data[self.data['Signal'] == -1].index, self.data['Close'][self.data['Signal'] == -1], 
                 'v', markersize=10, color='red', lw=0, label='Sell Signal')
        plt.title(f"{self.ticker} Moving Average Envelope Strategy")
        plt.xlabel('Date')
        plt.ylabel('Price')
        # Continue plotting Strategy 3
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}MovingAverageEnvelopeStrategy{PNG_SUFFIX}', format='png')
        plt.show()
        self.data.to_csv(f'{OUTPUT_DIR}MovingAverageEnvelopeStrategy{CSV_SUFFIX}')
    
    def multi_time_frame_breakout_strategy(self, period='5y', 
                                           weekly_rolling_window=WEEKLY_ROLLING_WINDOW, 
                                           volume_ma_period=VOLUME_MA_PERIOD):
        """Strategy 4: Multi-Time Frame Breakout."""
        # Fetch weekly data for resistance/support levels
        weekly_data = yf.download(self.ticker, period=period, interval='1wk')
        weekly_data = weekly_data[['Close']]
        weekly_data['Resistance'] = weekly_data['Close'].rolling(window=weekly_rolling_window).max()
        weekly_data['Support'] = weekly_data['Close'].rolling(window=weekly_rolling_window).min()
    
        # Fetch daily data for breakout detection
        daily_data = yf.download(self.ticker, period=period)
        daily_data = daily_data[['Close', 'Volume']]
        daily_data['Resistance'] = weekly_data['Resistance'].reindex(daily_data.index, method='ffill')
        daily_data['Support'] = weekly_data['Support'].reindex(daily_data.index, method='ffill')
        daily_data['Breakout'] = np.where(daily_data['Close'] > daily_data['Resistance'], 1,
                                          np.where(daily_data['Close'] < daily_data['Support'], -1, 0))
    
        # Confirm breakout with volume
        daily_data['Volume_MA'] = ta.SMA(daily_data['Volume'], timeperiod=volume_ma_period)
        daily_data['Volume_Confirm'] = np.where(daily_data['Volume'] > daily_data['Volume_MA'], 1, 0)
        daily_data['Final_Signal'] = daily_data['Breakout'] * daily_data['Volume_Confirm']
    
        # Plotting with consistent style
        plt.figure(figsize=(14, 12))
    
        # Plot Close Price with Weekly Support/Resistance & Buy/Sell Signals
        plt.subplot(3, 1, 1)
        plt.plot(daily_data['Close'], label='Close Price', color='blue', alpha=0.8)
        plt.plot(daily_data['Resistance'], label='Resistance (Weekly)', color='red', linestyle='--')
        plt.plot(daily_data['Support'], label='Support (Weekly)', color='green', linestyle='--')
        plt.plot(daily_data.loc[daily_data['Final_Signal'] == 1].index, 
                 daily_data['Close'][daily_data['Final_Signal'] == 1], '^', markersize=10, 
                 color='green', lw=0, label='Buy Signal')
        plt.plot(daily_data.loc[daily_data['Final_Signal'] == -1].index, 
                 daily_data['Close'][daily_data['Final_Signal'] == -1], 'v', markersize=10, 
                 color='red', lw=0, label='Sell Signal')
        plt.title('Close Price with Weekly Support/Resistance & Buy/Sell Signals')
        plt.legend(loc='best')
    
        # Plot Volume and Volume Moving Average
        plt.subplot(3, 1, 2)
        plt.plot(daily_data['Volume'], label='Volume', color='blue')
        plt.plot(daily_data['Volume_MA'], label=f'{volume_ma_period}-Day Volume MA', color='orange', linestyle='--')
        plt.title('Volume and 20-Day Moving Average')
        plt.legend(loc='best')
    
        # Plot Final Signal
        plt.subplot(3, 1, 3)
        plt.plot(daily_data['Final_Signal'], label='Final Signal', color='black')
        plt.axhline(1, color='green', linestyle='--', label='Buy Signal')
        plt.axhline(-1, color='red', linestyle='--', label='Sell Signal')
        plt.title('Final Trade Signal')
        plt.legend(loc='best')
    
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}MultiTimeFrameBreakoutStrategy{PNG_SUFFIX}', format='png')
        plt.show()
    
        # Save the resulting data to CSV
        daily_data.to_csv(f'{OUTPUT_DIR}MultiTimeFrameBreakoutStrategy{CSV_SUFFIX}')
    
        # Display the first few rows of the resulting data
        print(daily_data.head())
    def moving_average_envelope_strategy2(self, period='2y', 
                                     upper_envelope=UPPER_ENVELOPE, 
                                     lower_envelope=LOWER_ENVELOPE, 
                                     time_period_short=TIME_PERIOD_SHORT,
                                     trailing_stop_pct=0.05):
        """Strategy 3: Moving Average Envelope with Trailing Stop."""
        self.fetch_data(period)
        ema_string = f'EMA{time_period_short}'
        self.data[ema_string] = ta.EMA(self.data['Close'], timeperiod=time_period_short)
        self.data['UpperEnvelope'] = self.data[ema_string] * (1 + upper_envelope)
        self.data['LowerEnvelope'] = self.data[ema_string] * (1 - lower_envelope)
        self.data['Signal'] = 0  # Default to hold (0)
        
        # Buy signal: When price falls below the lower envelope
        self.data['Signal'][self.data['Close'] <= self.data['LowerEnvelope']] = 1
        
        # Initialize the trailing stop price column
        self.data['Trailing_Stop_Price'] = np.nan
        
        # Initialize flags for tracking position and stop price
        in_position = False
        trailing_stop_price = 0.0
    
        # Iterate over rows to set the trailing stop
        for i in range(1, len(self.data)):
            if self.data['Signal'][i] == 1:  # Buy signal
                in_position = True
                trailing_stop_price = self.data['Close'][i] * (1 - trailing_stop_pct)
                self.data.at[i, 'Trailing_Stop_Price'] = trailing_stop_price
            elif in_position:
                # Adjust trailing stop only if we are in a position
                current_price = self.data['Close'][i]
                # Update trailing stop price if the current price is above the previous stop
                if current_price > trailing_stop_price / (1 - trailing_stop_pct):
                    trailing_stop_price = current_price * (1 - trailing_stop_pct)
                self.data.at[i, 'Trailing_Stop_Price'] = trailing_stop_price
                
                # Check if the current price hits the trailing stop
                if current_price <= trailing_stop_price:
                    self.data.at[i, 'Signal'] = -1  # Sell signal
                    in_position = False  # Reset position
    
        # Plot with consistent style
        plt.plot(self.data['Close'], label='Close Price', color='blue', alpha=0.8)
        plt.plot(self.data[ema_string], label=f'{time_period_short}-Day EMA', color='orange')
        plt.plot(self.data['UpperEnvelope'], label=f'Upper Envelope ({upper_envelope*100:.1f}%)', color='red', linestyle='--')
        plt.plot(self.data['LowerEnvelope'], label=f'Lower Envelope ({lower_envelope*100:.1f}%)', color='green', linestyle='--')
        plt.plot(self.data[self.data['Signal'] == 1].index, self.data['Close'][self.data['Signal'] == 1], 
                 '^', markersize=10, color='green', lw=0, label='Buy Signal')
        plt.plot(self.data[self.data['Signal'] == -1].index, self.data['Close'][self.data['Signal'] == -1], 
                 'v', markersize=10, color='red', lw=0, label='Sell Signal')
        plt.plot(self.data['Trailing_Stop_Price'], label='Trailing Stop Price', linestyle='-.', color='purple')
        plt.title(f"{self.ticker} Moving Average Envelope Strategy with Trailing Stop")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}MovingAverageEnvelopeStrategy{PNG_SUFFIX}', format='png')
        plt.show()
        self.data.to_csv(f'{OUTPUT_DIR}MovingAverageEnvelopeStrategy{CSV_SUFFIX}')
    
       