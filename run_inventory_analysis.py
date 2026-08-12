import os
import pandas as pd

MASTER_FILE = "data/master/category_master_data.csv"  
FORECAST_FOLDER = "data/processed/"         
OUTPUT_FOLDER = "results/"                  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

from optimization.abc_analysis import perform_abc_analysis
from optimization.eoq_model import calculate_eoq_for_categories
from optimization.rop_model import calculate_safety_stock_and_rop
from optimization.savings_analysis import analyze_potential_savings
from optimization.sensitivity_analysis import sensitivity_analysis_comprehensive

# Read master file
master_df = pd.read_csv(MASTER_FILE)

# Process each forecast file
for forecast_file in os.listdir(FORECAST_FOLDER):
    if forecast_file.endswith(".csv"):
        forecast_path = os.path.join(FORECAST_FOLDER, forecast_file)
        forecast_df = pd.read_csv(forecast_path)

        # Merge forecast with master
        merged_df = pd.merge(forecast_df, master_df, on='Category', how='left')

        print(f"\nRunning analysis for forecast: {forecast_file}")

        # 1. ABC Analysis
        abc_results = perform_abc_analysis(merged_df)

        # 2. EOQ Calculation
        eoq_results = calculate_eoq_for_categories(merged_df, abc_results)
        eoq_file = os.path.join(OUTPUT_FOLDER, f"eoq_{forecast_file}")
        eoq_results.to_csv(eoq_file, index=False, encoding='utf-8-sig')

        # 3. ROP Calculation
        rop_results = calculate_safety_stock_and_rop(merged_df, eoq_results)
        rop_file = os.path.join(OUTPUT_FOLDER, f"rop_{forecast_file}")
        rop_results.to_csv(rop_file, index=False, encoding='utf-8-sig')

        # 4. Potential Savings Analysis
        savings_file = os.path.join(OUTPUT_FOLDER, f"savings_{forecast_file}")
        analyze_potential_savings(eoq_results, save_csv=True, output_path=savings_file)

        # 5. Sensitivity Analysis
        sensitivity_file = os.path.join(OUTPUT_FOLDER, f"sensitivity_{forecast_file}")
        sensitivity_analysis_comprehensive(eoq_results, rop_results, save_csv=True, output_path=sensitivity_file)

        print(f"Completed analysis for {forecast_file}. All results saved in {OUTPUT_FOLDER}")
