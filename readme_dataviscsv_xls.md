# Data Transformation Tool

A Python-based GUI application for data analysis and transformation, featuring interactive visualizations and statistical analysis capabilities.

## Features

- **Multiple Data Import Options**
  - Load CSV files
  - Load Excel files (multiple sheets supported)
  - Use example data for testing

- **Data Transformations**
  - Normalization (Min-Max scaling)
  - Standardization (Z-score)
  - Log transformation

- **Visual Analysis**
  - Interactive histograms with kernel density estimation
  - Statistical indicators overlay (mean, median, mode)
  - Quartile visualization
  - Standard deviation bounds

- **Statistical Analysis**
  - Comprehensive statistical metrics
  - Side-by-side comparison of original vs transformed data
  - Real-time updates

## Requirements

```
python 3.11 +
numpy 2.1.3
pandas 2.2.3
matplotlib 3.9.2
scipy 1.14.1
openpyxl 2.0.0
tkinter (usually comes with Python)
```

## Installation

1. Clone this repository or download the source code
2. Install the required packages:

```bash
pip install numpy pandas matplotlib scipy openpyxl
```

## Usage

1. Run the application:

```bash
python dataviscsv_xls.py
```

2. Load your data:
   - Click "Load CSV File" for CSV data
   - Click "Load Excel File" for Excel data
   - Click "Use Example Data" to test the application

3. Select a column to analyze from the dropdown menu


4. Apply transformations:
   - Click "Normalize" for min-max scaling (0-1 range)
   - Click "Standardize" for z-score standardization
   - Click "Log Transform" for logarithmic transformation

## Features in Detail

### Data Preview
- Shows the first 5 rows of your dataset
- Automatically handles date formatting
- Supports numeric and categorical data

### Transformations
- **Normalization**: Scales data to [0,1] range
- **Standardization**: Transforms data to have mean=0 and std=1
- **Log Transform**: Applies log10 transformation (requires positive values)

### Visualization
- Histogram with kernel density estimation
- Statistical markers:
  - Mean (red dashed line)
  - Median (blue solid line)
  - Mode (purple dotted line)
  - Standard deviation bounds (orange dashed lines)
  - Quartiles (green dotted lines)

### Statistics Panel
Displays key metrics for both original and transformed data:
- Mean
- Standard deviation
- Minimum and maximum values
- First and third quartiles

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
