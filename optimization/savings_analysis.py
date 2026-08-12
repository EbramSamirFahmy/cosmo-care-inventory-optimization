import pandas as pd
import os

def analyze_potential_savings(eoq_results, current_policy_orders_per_year=12, save_csv=True, output_path=None):
    """
    Compare EOQ system with a fixed-order policy (e.g., monthly orders)
    Args:
        eoq_results (pd.DataFrame): EOQ results for a given forecast
        current_policy_orders_per_year (int): number of orders per year in current policy
        save_csv (bool): if True, saves analysis to CSV
        output_path (str): path to save CSV if save_csv=True
    Returns:
        pd.DataFrame: savings analysis results
    """

    # Ensure results folder exists
    if save_csv:
        os.makedirs("results", exist_ok=True)
        if not output_path:
            output_path = os.path.join("results", "potential_savings_analysis.csv")

    savings_analysis = eoq_results.copy()

    # Monthly order quantity under current policy
    savings_analysis['Monthly_Order_Quantity'] = savings_analysis['Annual_Demand'] / current_policy_orders_per_year

    # Costs under current policy
    savings_analysis['Current_Ordering_Cost'] = current_policy_orders_per_year * savings_analysis['Ordering_Cost']
    savings_analysis['Current_Holding_Cost'] = (savings_analysis['Monthly_Order_Quantity'] / 2) * savings_analysis['Holding_Cost_Per_Unit']
    savings_analysis['Current_Total_Cost'] = savings_analysis['Current_Ordering_Cost'] + savings_analysis['Current_Holding_Cost']

    # Calculate savings
    savings_analysis['Cost_Saving'] = savings_analysis['Current_Total_Cost'] - savings_analysis['Total_Cost']
    savings_analysis['Saving_Percentage'] = (savings_analysis['Cost_Saving'] / savings_analysis['Current_Total_Cost']) * 100

    # Print summary on screen
    print(" Potential Savings Analysis:")
    print("=" * 60)
    total_current_cost = savings_analysis['Current_Total_Cost'].sum()
    total_eoq_cost = savings_analysis['Total_Cost'].sum()
    total_saving = total_current_cost - total_eoq_cost
    total_saving_percentage = (total_saving / total_current_cost) * 100
    print(f"Total cost with monthly ordering: {total_current_cost:,.2f}")
    print(f"Total cost with EOQ system: {total_eoq_cost:,.2f}")
    print(f"Potential annual savings: {total_saving:,.2f}")
    print(f"Savings percentage: {total_saving_percentage:.2f}%")
    print("=" * 60)

    # Top 5 saving categories
    top_savers = savings_analysis.sort_values('Cost_Saving', ascending=False).head()
    print("\n Top 5 categories by savings:")
    print(top_savers[['Category', 'ABC_Class', 'Current_Total_Cost',
                      'Total_Cost', 'Cost_Saving', 'Saving_Percentage']].to_string(index=False))

    # Save full data to CSV if requested
    if save_csv:
        savings_analysis.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved full savings analysis to: {output_path}")

        # Also save top 5 separately
        top5_path = os.path.join("results", "top5_saving_categories.csv")
        top_savers.to_csv(top5_path, index=False, encoding='utf-8-sig')
        print(f"Saved top 5 saving categories to: {top5_path}")

    return savings_analysis
