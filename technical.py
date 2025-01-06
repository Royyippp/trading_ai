import pandas as pd
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df
        self.close_price = df['close']
        self.high_price = df['high']
        self.low_price = df['low']
        self.open_price = df['open']
        self.volume = df['tick_volume']

    def calculate_open(self):
        """
        Calculate Open Price and add to the DataFrame.
        """
        return self.open_price
    
    def calculate_high(self):
        """
        Calculate High Price and add to the DataFrame.
        """
        return self.high_price
    
    def calculate_low(self):
        """
        Calculate Low Price and add to the DataFrame.
        """
        return self.low_price
    
    def calculate_close(self):
        """
        Calculate Close Price and add to the DataFrame.
        """
        return self.close_price

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Calculate MACD indicators and add to the DataFrame.
        """
        macd = self.close_price.ewm(span=fast).mean() - self.close_price.ewm(span=slow).mean()
        signal_line = macd.ewm(span=signal).mean()
        self.df['MACD'] = macd
        self.df['Signal'] = signal_line
        return {"MACD": self.df['MACD'], "Signal": self.df['Signal']}

    def calculate_rsi(self, length=14):
        """
        Calculate RSI and add to the DataFrame.
        """
        delta = self.close_price.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=length, min_periods=1).mean()
        avg_loss = loss.rolling(window=length, min_periods=1).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return {"RSI": self.df['RSI']}

    def calculate_sma(self, length=14):
        """
        Calculate Simple Moving Average (SMA) and add to the DataFrame.
        """
        self.df[f'SMA_{length}'] = self.close_price.rolling(window=length).mean()
        return {f"SMA_{length}": self.df[f'SMA_{length}']}

    def calculate_ema(self, length=14):
        """
        Calculate Exponential Moving Average (EMA) and add to the DataFrame.
        """
        self.df[f'EMA_{length}'] = self.close_price.ewm(span=length, adjust=False).mean()
        return {f"EMA_{length}": self.df[f'EMA_{length}']}

    def calculate_bollinger_bands(self, length=20, num_std_dev=2):
        """
        Calculate Bollinger Bands and add to the DataFrame.
        """
        sma = self.close_price.rolling(window=length).mean()
        rolling_std = self.close_price.rolling(window=length).std()
        self.df['BB_Upper'] = sma + (rolling_std * num_std_dev)
        self.df['BB_Lower'] = sma - (rolling_std * num_std_dev)
        self.df['BB_Middle'] = sma
        return {
            "BB_Upper": self.df['BB_Upper'],
            "BB_Lower": self.df['BB_Lower'],
            "BB_Middle": self.df['BB_Middle']
        }

    def calculate_atr(self, length=14):
        """
        Calculate Average True Range (ATR) and add to the DataFrame.
        """
        high_low = self.high_price - self.low_price
        high_close = abs(self.high_price - self.close_price.shift())
        low_close = abs(self.low_price - self.close_price.shift())
        true_range = high_low.combine(high_close, max).combine(low_close, max)
        self.df['ATR'] = true_range.rolling(window=length).mean()
        return {"ATR": self.df['ATR']}

    def calculate_adx(self, length=14):
        """
        Calculate Average Directional Index (ADX) and add to the DataFrame.
        """
        high_diff = self.high_price.diff()
        low_diff = self.low_price.diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        atr = self.calculate_atr(length)['ATR']
        plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        self.df['ADX'] = dx.rolling(window=length).mean()
        return {"ADX": self.df['ADX']}

    def calculate_stochastic_oscillator(self, length=14):
        """
        Calculate Stochastic Oscillator and add to the DataFrame.
        """
        low_min = self.low_price.rolling(window=length).min()
        high_max = self.high_price.rolling(window=length).max()
        self.df['Stoch_Oscillator'] = 100 * (self.close_price - low_min) / (high_max - low_min)
        return {"Stoch_Oscillator": self.df['Stoch_Oscillator']}

    def calculate_momentum(self, length=14):
        """
        Calculate Momentum Indicator and add to the DataFrame.
        """
        self.df['Momentum'] = self.close_price.diff(periods=length)
        return {"Momentum": self.df['Momentum']}