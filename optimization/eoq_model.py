
import pandas as pd
import numpy as np

def calculate_eoq_for_categories(data, abc_results):
    """
    Calculate EOQ for each category while considering ABC classification
    """

    annual_demand = (
        data.groupby('Category')['Forecast_Demand']
        .sum()
        .reset_index(name='Annual_Demand')
    )

    product_info = data[
        ['Category', 'Cost_Price', 'Lead_Time',
         'Ordering_Cost', 'Holding_Cost_Percent',
         'Initial_Stock']
    ].drop_duplicates()

    eoq_data = pd.merge(annual_demand, product_info, on='Category')

    eoq_data = pd.merge(
        eoq_data,
        abc_results[['Category', 'ABC_Class']],
        on='Category'
    )

    results = []

    for _, row in eoq_data.iterrows():
        H = row['Cost_Price'] * (row['Holding_Cost_Percent'] / 100)
        D = row['Annual_Demand']
        S = row['Ordering_Cost']

        if H > 0 and D > 0:
            eoq = np.sqrt((2 * D * S) / H)
            order_frequency = D / eoq
            order_cycle_days = 365 / order_frequency
            total_ordering_cost = (D / eoq) * S
            total_holding_cost = (eoq / 2) * H
            total_cost = total_ordering_cost + total_holding_cost
        else:
            eoq = order_frequency = order_cycle_days = 0
            total_ordering_cost = total_holding_cost = total_cost = 0

        results.append({
            'Category': row['Category'],
            'ABC_Class': row['ABC_Class'],
            'Annual_Demand': D,
            'Unit_Cost': row['Cost_Price'],
            'Ordering_Cost': S,
            'Holding_Cost_Percent': row['Holding_Cost_Percent'],
            'Holding_Cost_Per_Unit': H,
            'EOQ': round(eoq, 2),
            'Order_Frequency': round(order_frequency, 2),
            'Order_Cycle_Days': round(order_cycle_days, 2),
            'Total_Ordering_Cost': round(total_ordering_cost, 2),
            'Total_Holding_Cost': round(total_holding_cost, 2),
            'Total_Cost': round(total_cost, 2),
            'Lead_Time_Days': row['Lead_Time'],
            'Initial_Stock': row['Initial_Stock']
        })

    return pd.DataFrame(results)
