import numpy as np

def dixon_test(data, alpha=0.05):
        """
            Perform Dixon's Q test for outliers.
            Parameters:
                data (array-like): 1D array of numeric data.
                alpha (float): Significance level (default is 0.05).
            Returns:
                outliers (list): List of detected outliers.
        """
        data = np.sort(data)
        n = len(data)

        if n < 3 or n > 30:
                raise ValueError("Dixon's test is only valid for sample sizes between 3 and 30.")

                # Q critical values table for alpha=0.05
        q_critical = {
                3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507, 8: 0.468,
                9: 0.437, 10: 0.412, 11: 0.392, 12: 0.376, 13: 0.361, 14: 0.349,
                15: 0.338, 16: 0.329, 17: 0.320, 18: 0.313, 19: 0.306, 20: 0.300,
                21: 0.295, 22: 0.290, 23: 0.285, 24: 0.281, 25: 0.277, 26: 0.273,
                27: 0.270, 28: 0.267, 29: 0.263, 30: 0.260
            }

        q_min = (data[1] - data[0]) / (data[-1] - data[0])
        q_max = (data[-1] - data[-2]) / (data[-1] - data[0])

        outliers = []
        if q_min > q_critical[n]:
            outliers.append(data[0])
        if q_max > q_critical[n]:
            outliers.append(data[-1])

        return outliers