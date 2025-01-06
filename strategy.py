import pandas as pd

class Strategy:
    def __init__(self, data, entry_conditions, exit_conditions):
        """
        Initialize the Strategy object.

        Parameters:
        - data: pd.DataFrame, the historical price data.
        - entry_conditions: list of Condition objects, conditions to enter a trade.
        - exit_conditions: list of Condition objects, conditions to exit a trade.
        """
        self.data = data
        self.entry_conditions = entry_conditions
        self.exit_conditions = exit_conditions

    def main(self):
        """
        Backtest the strategy using the entry and exit conditions.

        Returns:
        - results: dict, including win percentage, total trades, equity curve, and other metrics.
        """
        # Generate entry and exit signals
        entry_signal = self._combine_conditions(self.entry_conditions)
        exit_signal = self._combine_conditions(self.exit_conditions)

        # Initialize backtesting variables
        position = None  # Track if we are in a trade
        entry_price = 0
        trades = []
        equity_curve = [1]  # Start with an initial equity of 1
        trade_entries = []
        trade_exits = []

        # Iterate through the dataset
        for i in range(len(self.data)):
            if entry_signal.iloc[i] and position is None:
                # Enter a trade
                position = 'long'
                entry_price = self.data['close'].iloc[i]
                trade_entries.append((self.data['time'].iloc[i], entry_price))

            elif exit_signal.iloc[i] and position == 'long':
                # Exit the trade
                exit_price = self.data['close'].iloc[i]
                profit = exit_price - entry_price
                trades.append(profit > 0)  # True if profit, False if loss
                position = None
                trade_exits.append((self.data['time'].iloc[i], exit_price))

                # Update equity curve
                equity_curve.append(equity_curve[-1] * (1 + profit / entry_price))

            else:
                # If no trade is entered or exited, equity remains the same
                equity_curve.append(equity_curve[-1])

        # Calculate metrics
        total_trades = len(trades)
        win_trades = sum(trades)
        win_percentage = (win_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            "win_percentage": win_percentage,
            "total_trades": total_trades,
            "winning_trades": win_trades,
            "losing_trades": total_trades - win_trades,
            "equity_curve": equity_curve,
            "trade_entries": trade_entries,
            "trade_exits": trade_exits
        }

    def _combine_conditions(self, conditions):
        """
        Combine multiple conditions using logical AND.

        Parameters:
        - conditions: list of Condition objects.

        Returns:
        - combined_series: pd.Series, the combined Boolean Series.
        """
        if not conditions:
            raise ValueError("No conditions provided for combination.")
        
        combined_series = pd.Series(True, index=self.data.index)

        for idx, condition in enumerate(conditions):
            condition_result = condition.generate()
            print(f"Condition {idx + 1}:\n{condition_result.head()}")  # Debug each condition
            combined_series &= condition_result

        return combined_series