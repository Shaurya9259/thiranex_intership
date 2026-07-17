# Data Cleaning & Visualization Project

Internship task: work on a raw dataset to clean, process, and visualize insights.

## Overview
This project takes a raw, messy retail sales dataset (1,200+ orders) and:
1. Cleans it — handles missing values, duplicates, outliers, and inconsistent formatting
2. Processes it — standardizes types, parses mixed date formats, engineers derived columns
3. Visualizes it — builds a summary dashboard and supporting charts

## Project Structure
```
data-cleaning-viz-project/
├── data/
│   ├── generate_raw_data.py     # Creates the synthetic raw dataset
│   ├── raw_sales_data.csv       # Raw, messy dataset (input)
│   └── clean_sales_data.csv     # Cleaned dataset (output)
├── notebooks/
│   ├── 01_data_cleaning.py      # Cleaning & preprocessing pipeline
│   └── 02_visualization.py      # Visualization / dashboard generation
├── outputs/
│   ├── dashboard.png            # Combined 4-chart dashboard
│   ├── unit_price_distribution.png
│   ├── correlation_heatmap.png
│   └── category_summary.csv
├── requirements.txt
└── README.md
```

## Data Issues Handled
| Issue | How it was handled |
|---|---|
| Missing values (`unit_price`, `units_sold`, `region`, `payment_mode`) | Numeric columns filled with category-wise median; categorical columns filled with mode |
| Duplicate rows | Detected and removed via `order_id`, keeping first occurrence |
| Outliers in `units_sold` / `unit_price` | Detected with the IQR method and capped at the outlier boundary |
| Inconsistent text casing/whitespace (`" ELECTRONICS "`, `"electronics"`) | Stripped and standardized to title case |
| Mixed date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-Mon-YYYY`, `MM/DD/YYYY`) | Parsed with format-by-format matching into a single `datetime64` column |

## Key Insights
- Revenue is fairly evenly spread across categories, with **Books** and **Home & Kitchen** leading
- Monthly revenue shows [seasonal fluctuation — see `dashboard.png`]
- **Electronics** has the highest order count but lowest average order value
- `units_sold`, `unit_price`, and `revenue` correlations are shown in `correlation_heatmap.png`

## How to Run
```bash
pip install -r requirements.txt
python data/generate_raw_data.py      # (optional) regenerate raw data
python notebooks/01_data_cleaning.py  # clean the data
python notebooks/02_visualization.py  # generate charts
```

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn

## Expected Outcome
This project demonstrates a full data preprocessing and visualization workflow: identifying data quality issues, applying appropriate cleaning strategies, and communicating findings through clear visual storytelling.
