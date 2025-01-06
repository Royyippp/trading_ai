import pandas as pd

class Condition:
    def __init__(self, series_a, operator, series_b):
        """
        Initialize the Condition object.

        Parameters:
        - series_a: pd.Series, the first series for comparison.
        - operator: str, the comparison operator ('>', '<', '==', '>=', '<=', '!=', 'crosses_above', 'crosses_below').
        - series_b: pd.Series or scalar, the second series for comparison or a scalar value.
                   If a scalar value is provided, it will be converted to a pd.Series of the same length as series_a.
        """
        self.series_a = series_a
        self.operator = operator

        # If series_b is a scalar, create a pd.Series with the same length as series_a
        if isinstance(series_b, (int, float)):
            self.series_b = pd.Series([series_b] * len(series_a), index=series_a.index)
        elif isinstance(series_b, pd.Series):
            self.series_b = series_b
        else:
            raise ValueError("series_b must be a scalar (int/float) or a pd.Series")

    def generate(self):
        """
        Generate a Boolean pd.Series based on the specified condition.

        Returns:
        - pd.Series: A Boolean series representing the result of the condition.
        """
        print(f"Generating condition with operator '{self.operator}'")
        print(f"Type of series_a: {type(self.series_a)}, Type of series_b: {type(self.series_b)}")

        if self.operator == '>':
            return self.series_a > self.series_b
        elif self.operator == '<':
            return self.series_a < self.series_b
        elif self.operator == '==':
            return self.series_a == self.series_b
        elif self.operator == '>=':
            return self.series_a >= self.series_b
        elif self.operator == '<=':
            return self.series_a <= self.series_b
        elif self.operator == '!=':
            return self.series_a != self.series_b
        elif self.operator == 'crosses_above':
            return (self.series_a.shift(1) <= self.series_b.shift(1)) & (self.series_a > self.series_b)
        elif self.operator == 'crosses_below':
            return (self.series_a.shift(1) >= self.series_b.shift(1)) & (self.series_a < self.series_b)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")