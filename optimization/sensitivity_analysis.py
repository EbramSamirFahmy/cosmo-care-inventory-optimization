import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt

def sensitivity_analysis_comprehensive(eoq_results, rop_results, base_scenario="current", save_csv=True, output_path=None):
    """
    Comprehensive sensitivity analysis for the inventory system
    Args:
        eoq_results (pd.DataFrame): EOQ results for a forecast
        rop_results (pd.DataFrame): ROP results for the same forecast
        base_scenario (str): scenario name
        save_csv (bool): if True, save results to CSV
        output_path (str): CSV path if save_csv=True
    Returns:
        pd.DataFrame: detailed sensitivity analysis
    """


    # Scenarios
    scenarios = {
        'demand_increase_20': {'demand_multiplier': 1.2, 'name': '+20% Demand'},
        'demand_decrease_20': {'demand_multiplier': 0.8, 'name': '-20% Demand'},
        'holding_cost_increase_30': {'holding_multiplier': 1.3, 'name': '+30% Holding Cost'},
        'ordering_cost_decrease_25': {'ordering_multiplier': 0.75, 'name': '-25% Ordering Cost'},
        'lead_time_increase_50': {'lead_time_multiplier': 1.5, 'name': '+50% Lead Time'},
        'combined_optimistic': {
            'demand_multiplier': 1.1,
            'ordering_multiplier': 0.8,
            'holding_multiplier': 0.9,
            'lead_time_multiplier': 0.8,
            'name': 'Optimistic Scenario'
        },
        'combined_pessimistic': {
            'demand_multiplier': 0.9,
            'ordering_multiplier': 1.2,
            'holding_multiplier': 1.2,
            'lead_time_multiplier': 1.5,
            'name': 'Pessimistic Scenario'
        }
    }

    # Pick sample Category A
    category_A = eoq_results[eoq_results['ABC_Class'] == 'A']['Category'].unique()
    if len(category_A) == 0:
        return None

    sample_category = category_A[0]
    base_data = eoq_results[eoq_results['Category']==sample_category].iloc[0]
    base_rop = rop_results[rop_results['Category']==sample_category].iloc[0]

    results = []

    for params in scenarios.values():
        demand = base_data['Annual_Demand'] * params.get('demand_multiplier',1)
        ordering_cost = base_data['Ordering_Cost'] * params.get('ordering_multiplier',1)
        holding_cost = base_data['Holding_Cost_Per_Unit'] * params.get('holding_multiplier',1)
        lead_time = base_data['Lead_Time_Days'] * params.get('lead_time_multiplier',1)

        new_eoq = np.sqrt((2*demand*ordering_cost)/holding_cost) if holding_cost>0 else 0
        daily_demand = demand/365
        service_level = {'A':0.98,'B':0.95,'C':0.90}.get('A',0.95)
        z_score = stats.norm.ppf(service_level)
        daily_std_dev = daily_demand*0.3
        safety_stock = z_score*daily_std_dev*np.sqrt(lead_time)
        new_rop = daily_demand*lead_time + safety_stock

        new_ordering_cost = (demand/new_eoq)*ordering_cost if new_eoq>0 else 0
        new_holding_cost = (new_eoq/2)*holding_cost if new_eoq>0 else 0
        new_total_cost = new_ordering_cost + new_holding_cost

        results.append({
            'Scenario': params['name'],
            'New_EOQ': round(new_eoq,1),
            'EOQ_Change_%': round((new_eoq-base_data['EOQ'])/base_data['EOQ']*100,1),
            'New_ROP': round(new_rop,1),
            'ROP_Change_%': round((new_rop-base_rop['Reorder_Point'])/base_rop['Reorder_Point']*100,1),
            'New_Total_Cost': round(new_total_cost,1),
            'Total_Cost_Change_%': round((new_total_cost-base_data['Total_Cost'])/base_data['Total_Cost']*100,1),
            'Safety_Stock': round(safety_stock,1)
        })

    results_df = pd.DataFrame(results)

    if save_csv:
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return results_df
