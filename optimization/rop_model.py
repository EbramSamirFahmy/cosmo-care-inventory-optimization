
import pandas as pd
import numpy as np
from scipy import stats

def calculate_safety_stock_and_rop(data, eoq_results):
    """
    Calculate safety stock and reorder point (ROP)
    """

    monthly_stats = (
        data.groupby(['Category', 'Month'])['Forecast_Demand']
        .sum()
        .reset_index()
    )

    demand_volatility = (
        monthly_stats.groupby('Category')['Forecast_Demand']
        .std()
        .reset_index(name='Monthly_Std_Dev')
    )

    rop_data = pd.merge(eoq_results, demand_volatility, on='Category', how='left')

    rop_data['Daily_Demand'] = rop_data['Annual_Demand'] / 365
    rop_data['Daily_Std_Dev'] = rop_data['Monthly_Std_Dev'] / np.sqrt(30)

    service_levels = {'A': 0.98, 'B': 0.95, 'C': 0.90}

    results = []

    for _, row in rop_data.iterrows():
        service_level = service_levels.get(row['ABC_Class'], 0.95)
        z_score = stats.norm.ppf(service_level)

        if pd.notna(row['Daily_Std_Dev']):
            safety_stock = z_score * row['Daily_Std_Dev'] * np.sqrt(row['Lead_Time_Days'])
        else:
            safety_stock = z_score * row['Daily_Demand'] * 0.3 * np.sqrt(row['Lead_Time_Days'])

        demand_during_leadtime = row['Daily_Demand'] * row['Lead_Time_Days']
        reorder_point = demand_during_leadtime + safety_stock
        max_inventory_level = safety_stock + row['EOQ']

        current_stock = row['Initial_Stock']

        if current_stock > max_inventory_level:
            stock_status = 'Excess Stock'
        elif current_stock > reorder_point:
            stock_status = 'Normal'
        elif current_stock > safety_stock:
            stock_status = 'Monitor Closely'
        else:
            stock_status = 'URGENT: Need to Order'

        results.append({
            'Category': row['Category'],
            'ABC_Class': row['ABC_Class'],
            'Lead_Time_Days': row['Lead_Time_Days'],
            'Daily_Demand': round(row['Daily_Demand'], 2),
            'Daily_Std_Dev': round(row['Daily_Std_Dev'], 2) if pd.notna(row['Daily_Std_Dev']) else 0,
            'Service_Level': service_level,
            'Z_Score': round(z_score, 3),
            'Safety_Stock': round(max(safety_stock, 0), 2),
            'Demand_During_LeadTime': round(demand_during_leadtime, 2),
            'Reorder_Point': round(reorder_point, 2),
            'Max_Inventory_Level': round(max_inventory_level, 2),
            'EOQ': row['EOQ'],
            'Current_Stock': current_stock,
            'Stock_Status': stock_status,
            'Order_Urgency': 1 if stock_status == 'URGENT: Need to Order' else 0
        })

    return pd.DataFrame(results)
