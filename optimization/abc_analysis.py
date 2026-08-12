import pandas as pd

def perform_abc_analysis(data):
    """
    ABC analysis based on expected annual inventory value
    """

    category_summary = data.groupby('Category').agg({
        'Forecast_Demand': 'sum',
        'Cost_Price': 'first',
        'Selling_Price': 'first'
    }).reset_index()

    category_summary['Annual_Value'] = (
        category_summary['Forecast_Demand'] * category_summary['Cost_Price']
    )

    category_summary = category_summary.sort_values(
        'Annual_Value', ascending=False
    )

    category_summary['Cumulative_Value'] = category_summary['Annual_Value'].cumsum()
    total_value = category_summary['Annual_Value'].sum()

    category_summary['Cumulative_Percentage'] = (
        category_summary['Cumulative_Value'] / total_value
    ) * 100

    def assign_abc(cum_pct):
        if cum_pct <= 80:
            return 'A'
        elif cum_pct <= 95:
            return 'B'
        else:
            return 'C'

    category_summary['ABC_Class'] = (
        category_summary['Cumulative_Percentage'].apply(assign_abc)
    )

    return category_summary
