# Enterprise Cross-Sell Heatmap

An interactive visualization tool for analyzing cross-selling opportunities in banking products across Personal Banking and Business Banking segments.

## Features

- Dual matrix visualization for Personal Banking and Business Banking relationships
- Interactive heatmaps showing penetration rates
- Detailed cell annotations with current penetration, target penetration, and opportunity gaps
- Special overlap zone highlighting cross-segment products
- Responsive design with Bootstrap styling

## Setup

### Option 1: Automatic Setup (Recommended)

Run the setup script to create a virtual environment and install dependencies:
```bash
./setup.sh
```

### Option 2: Manual Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate  # On Windows
```

2. Run the application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

## Data Structure

The visualization currently uses sample data. To use real data, modify the `generate_sample_data` function in `app.py` with your actual penetration rates.

## Matrix Structure

### Personal Banking Matrix
- Anchor Products: CASA, Credit Cards, Mortgages, Personal Loans, Fixed Deposits, Wealth Management, Digital Banking
- Additional Cross-sell Products: Money Transfer Services, Bill Payments, Investment Products, Insurance Products

### Business Banking Matrix
- Anchor Products: Business CASA, Business Loans, Trade Working Capital, Asset Based Finance, Digital Banking (Business)
- Additional Cross-sell Products: FX Services, Payment Solutions, Business Insurance

### Overlap Zone
Products that are relevant across both segments:
- Digital Banking
- FX Services
- Payment Solutions
- Basic Investment Products