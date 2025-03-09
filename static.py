import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import cost_cosnsolidation
import calendar
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyecharts import options as opts
from pyecharts.charts import Calendar, Page
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components

consolidations_day_mapping = {
    # 5-day scenarios
    'Mon_Tue_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Tue_Wed_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Fri_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },

    # 4-day scenarios
    'Mon_Tue_Wed_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Wed_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Thu_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },

    # 3-day scenarios
    'Mon_Tue_Wed': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Tue_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Mon_Wed_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Wed_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Tue_Wed_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Wed_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 2-day scenarios
    'Mon_Tue': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Mon_Wed': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': 0
    },
    'Tue_Wed': {
        'Mon': -5,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Tue_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Wed_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Wed_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Wed_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Wed_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 1-day scenarios
    'Only_Mon': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': -6
    },
    'Only_Tue': {
        'Mon': -6,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Only_Wed': {
        'Mon': -5,
        'Tue': -6,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Only_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': -6,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Only_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': -5,
        'Thu': -6,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Only_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': -4,
        'Thu': -5,
        'Fri': -6,
        'Sat': 0,
        'Sun': -1

    },
    'Only_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': -3,
        'Thu': -4,
        'Fri': -5,
        'Sat': -6,
        'Sun': 0

    }
}

def create_consolidated_shipments_calendar(consolidated_df):
    # Group by UPDATED_DATE and calculate consolidated_shipment_cost and Total Pallets
    df_consolidated = consolidated_df.groupby('UPDATED_DATE').agg({
        'consolidated_shipment_cost': 'sum',  # Sum of shipment costs
        'Total Pallets': 'sum'  # Sum of total pallets
    }).reset_index()
    df_consolidated.columns = ['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']

    # Split data by year
    df_2023 = df_consolidated[df_consolidated['UPDATED_DATE'].dt.year == 2023]
    df_2024 = df_consolidated[df_consolidated['UPDATED_DATE'].dt.year == 2024]

    calendar_data_2023 = df_2023[['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']].values.tolist()
    calendar_data_2024 = df_2024[['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"Calendar Heatmap for Consolidated Shipments ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[2] for item in data) if data else 0,  # Use Total Pallets for heatmap
                    min_=min(item[2] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + 
                                   '<br/>Consolidated Shipment Cost: ' + p.data[1] +
                                   '<br/>Total Pallets: ' + p.data[2];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)

    return calendar_2023, calendar_2024


def create_original_orders_calendar(original_df):
    # Group by SHIPPED_DATE and calculate the number of orders and total pallets
    df_original = original_df.groupby('SHIPPED_DATE').agg({
        'ORDER_ID': 'count',  # Count of orders
        'Total Pallets': 'sum'  # Sum of total pallets
    }).reset_index()
    df_original.columns = ['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']

    # Split data by year
    df_2023 = df_original[df_original['SHIPPED_DATE'].dt.year == 2023]
    df_2024 = df_original[df_original['SHIPPED_DATE'].dt.year == 2024]

    calendar_data_2023 = df_2023[['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']].values.tolist()
    calendar_data_2024 = df_2024[['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Original Orders ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[1] for item in data) if data else 0,  # Use Orders Shipped for heatmap
                    min_=min(item[1] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + 
                                   '<br/>Orders Shipped: ' + p.data[1] +
                                   '<br/>Total Pallets: ' + p.data[2];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)

    return calendar_2023, calendar_2024


def create_heatmap_and_bar_charts(consolidated_df, original_df, start_date, end_date):
    # Create calendar charts (existing code)
    chart_original_2023, chart_original_2024 = create_original_orders_calendar(original_df)
    chart_consolidated_2023, chart_consolidated_2024 = create_consolidated_shipments_calendar(consolidated_df)

    # Create bar charts for orders over time
    def create_bar_charts(df_original, df_consolidated, year):
        # Filter data for the specific year
        mask_original = df_original['SHIPPED_DATE'].dt.year == year
        year_data_original = df_original[mask_original]

        # For consolidated data
        if 'Date' in df_consolidated.columns:
            mask_consolidated = pd.to_datetime(df_consolidated['Date']).dt.year == year
            year_data_consolidated = df_consolidated[mask_consolidated]
        else:
            year_data_consolidated = pd.DataFrame()

        # Create subplot figure with shared x-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'Daily Orders Before Consolidation ({year})',
                f'Daily Orders After Consolidation ({year})'
            )
        )

        # Add bar chart for original orders
        if not year_data_original.empty:
            daily_orders = year_data_original.groupby('SHIPPED_DATE').size().reset_index()
            daily_orders.columns = ['Date', 'Orders']

            fig.add_trace(
                go.Bar(
                    x=daily_orders['Date'],
                    y=daily_orders['Orders'],
                    name='Orders',
                    marker_color='#1f77b4'
                ),
                row=1,
                col=1
            )

        # Add bar chart for consolidated orders
        if not year_data_consolidated.empty:
            daily_consolidated = year_data_consolidated.groupby('Date').agg({
                'Orders': lambda x: sum(len(orders) for orders in x)
            }).reset_index()

            fig.add_trace(
                go.Bar(
                    x=daily_consolidated['Date'],
                    y=daily_consolidated['Orders'],
                    name='Orders',
                    marker_color='#749f77'
                ),
                row=2,
                col=1
            )

        # Update layout
        fig.update_layout(
            height=500,  # Increased height for better visibility
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=20, t=60, b=20),
            hovermode='x unified'
        )

        # Update x-axes
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                thickness=0.05,  # Make the rangeslider thinner
                bgcolor='#F4F4F4',  # Light gray background
                bordercolor='#DEDEDE',  # Slightly darker border
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(
            rangeslider=dict(visible=False),
            row=1,
            col=1
        )

        # Update y-axes
        fig.update_yaxes(title_text="Number of Orders", row=1, col=1)
        fig.update_yaxes(title_text="Number of Orders", row=2, col=1)

        return fig

    # Create bar charts for both years
    bar_charts_2023 = create_bar_charts(original_df, consolidated_df, 2023)
    bar_charts_2024 = create_bar_charts(original_df, consolidated_df, 2024)

    return {
        2023: (chart_original_2023, chart_consolidated_2023, bar_charts_2023),
        2024: (chart_original_2024, chart_consolidated_2024, bar_charts_2024)
    }


# Load data from files
complete_input_path = 'Complete Input.xlsx'
rate_card_path = 'Cost per pallet.xlsx'

def load_data(complete_input_path, rate_card_path):
    """
    Load data from Excel files into pandas DataFrames.
    """
    try:
        complete_input = pd.read_excel(complete_input_path)
        rate_card = pd.read_excel(rate_card_path)
        return complete_input, rate_card
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

complete_input, rate_card = load_data(complete_input_path, rate_card_path)

def calculate_cost(total_pallets, prod_type, postcode, rate_card):
    """
    Calculate the cost based on the number of pallets, product type, and postcode.
    """

    # Filter the rate card based on product type and postcode
    filtered_rate = rate_card[(rate_card['PROD TYPE'] == prod_type) &
                              (rate_card['SHORT_POSTCODE'] == postcode)]

    if filtered_rate.empty:
        return None

    # Initialize total cost
    total_cost = 0

    # Handle cost calculation
    if total_pallets <= 52:
        cost_column = f"COST PER {int(total_pallets)} PALLET{'S' if total_pallets > 1 else ''}"
        if cost_column in filtered_rate.columns:
            total_cost = filtered_rate[cost_column].values[0]
    else:
        # Split into full batches of 52 pallets and remaining pallets
        full_batches = total_pallets // 52
        remaining_pallets = total_pallets % 52

        # Calculate cost for full batches
        batch_column = "COST PER 52 PALLETS"
        if batch_column in filtered_rate.columns:
            total_cost += full_batches * filtered_rate[batch_column].values[0]

        # Calculate cost for remaining pallets
        if remaining_pallets > 0:
            remaining_column = f"COST PER {int(remaining_pallets)} PALLET{'S' if remaining_pallets > 1 else ''}"
            if remaining_column in filtered_rate.columns:
                total_cost += filtered_rate[remaining_column].values[0]

    # Return the total cost
    return total_cost if total_cost > 0 else None



def cost_of_columns(filtered_data, rate_card):
    print(filtered_data)
    aggregated_data = filtered_data.groupby(
        ['PROD TYPE', 'SHORT_POSTCODE', 'ORDER_ID', 'SHIPPED_DATE'], as_index=False
    ).agg({'Total Pallets': 'sum', 'Distance': 'first', 'NAME': 'first'})


    aggregated_data['shipment_cost'] = aggregated_data.apply(
        lambda row: calculate_cost(row['Total Pallets'], row['PROD TYPE'], row['SHORT_POSTCODE'], rate_card),
        axis=1
    )
    return aggregated_data , aggregated_data['shipment_cost'].sum()



def get_updated_delivery_date(current_date, day_mapping):
    current_day = current_date.strftime('%a')
    updated_date = current_date + timedelta(day_mapping.get(current_day, 0))
    return updated_date


def consolidate_shipments(aggregated_data, rate_card, day_mapping):

    aggregated_data['UPDATED_DATE'] = aggregated_data['SHIPPED_DATE'].apply(
        lambda x: get_updated_delivery_date(x, day_mapping)
    )

    consolidated_data = aggregated_data.groupby(
        ['PROD TYPE', 'SHORT_POSTCODE', 'UPDATED_DATE'], as_index=False
    ).agg({'Total Pallets': 'sum', 'Distance': 'first', 'NAME': 'first'})

    # Calculate the consolidated shipment cost
    consolidated_data['consolidated_shipment_cost'] = consolidated_data.apply(
        lambda row: calculate_cost(row['Total Pallets'], row['PROD TYPE'], row['SHORT_POSTCODE'], rate_card),
        axis=1
    )
    return consolidated_data ,consolidated_data['consolidated_shipment_cost'].sum()



# Check if data was loaded successfully
def find_cost_savings(complete_input, rate_card, selected_scenarios ,parameters):


    filtered_data = cost_cosnsolidation.get_filtered_data(parameters, complete_input)


    aggregated_data , total_shipment_cost = cost_of_columns(filtered_data, rate_card)
    # st.dataframe(aggregated_data)

    scenario_results = []
    for scenario in selected_scenarios:
        day_mapping = consolidations_day_mapping[scenario]
        consolidated_data, total_consolidated_cost = consolidate_shipments(aggregated_data, rate_card, day_mapping)

        scenario_results.append({
            'scenario': scenario,
            'total_consolidated_cost': total_consolidated_cost,
            'num_shipments': len(consolidated_data.index),
            'avg_pallets': round(consolidated_data['Total Pallets'].mean(), 2)
        })

        # Step 5: Determine the best scenario
    best_scenario = min(scenario_results, key=lambda x: x['total_consolidated_cost'])

    # Step 6: Display results using Streamlit
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write('**Before Consolidation:**')
    #     st.write(f"Total Shipment Cost: **€{total_shipment_cost:,.2f}**")
    #     st.write(f"No of shipments: {len(aggregated_data.index):,}")
    #     st.write(f"Avg Pallets: {round(aggregated_data['Total Pallets'].mean(), 2)}")
    #
    # with col2:
    #     st.write('**After Consolidation:**')
    #     st.write('**Best Cost Savings Scenario:**')
    #     st.write(f"Delivery Scenario: **{best_scenario['scenario']}**")
    #     st.write(f"Total Consolidated Shipment Cost: **€{best_scenario['total_consolidated_cost']:,.2f}**")
    #     st.write(f"No of shipments: {best_scenario['num_shipments']:,}")
    #     st.write(f"Avg Pallets: {best_scenario['avg_pallets']}")

    metric_style = """
        <div style="
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        ">
            <span style="font-weight: bold;">{label}:</span> {value}
        </div>
    """

    change_style = """
        <div style="
            background-color: #e8f0fe;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="font-weight: bold;">{label}:</span>
            <span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>
        </div>
    """

    # Create columns
    st.write(f"**For the Best cost saving Scenario : {best_scenario['scenario']} ⬇️**")
    col1, col2 , col3 = st.columns(3)

    # Before Consolidation
    with col1:
        st.markdown("##### Before Consolidation")
        st.markdown(metric_style.format(
            label="Total Shipment Cost",
            value=f"€{total_shipment_cost:,.2f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="No of Shipments",
            value=f"{len(aggregated_data.index):,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Avg Pallets",
            value=f"{round(aggregated_data['Total Pallets'].mean(), 2)}"
        ), unsafe_allow_html=True)

    # After Consolidation
    with col2:
        st.markdown(f"##### After Consolidation")
        # st.markdown("**Best Cost Savings Scenario:**", unsafe_allow_html=True)
        # st.markdown(metric_style.format(
        #     label="Delivery Scenario",
        #     value=f"{best_scenario['scenario']}"
        # ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Total Consolidated Shipment Cost",
            value=f"€{best_scenario['total_consolidated_cost']:,.2f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="No of Shipments",
            value=f"{best_scenario['num_shipments']:,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Avg Pallets",
            value=f"{best_scenario['avg_pallets']}"
        ), unsafe_allow_html=True)

    # Percentage Changes
    with col3:
        st.markdown("##### Percentage Change")

        # Calculate percentage changes
        shipment_cost_change = ((best_scenario[
                                     'total_consolidated_cost'] - total_shipment_cost) / total_shipment_cost) * 100
        num_shipments_change = ((best_scenario['num_shipments'] - len(aggregated_data.index)) / len(
            aggregated_data.index)) * 100
        avg_pallets_change = ((best_scenario['avg_pallets'] - aggregated_data['Total Pallets'].mean()) /
                              aggregated_data['Total Pallets'].mean()) * 100

        # Display percentage changes
        st.markdown(change_style.format(
            label="Shipment Cost",
            value=shipment_cost_change,
            color="green" if shipment_cost_change < 0 else "red"  # Negative change is good (cost savings)
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="No of Shipments",
            value=num_shipments_change,
            color="green" if num_shipments_change < 0 else "red"  # Negative change is good (fewer shipments)
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="Avg Pallets",
            value=avg_pallets_change,
            color="green" if avg_pallets_change > 0 else "red"  # Positive change is good (more pallets per shipment)
        ), unsafe_allow_html=True)

    st.write(" ")
    st.write(" ")

    # Step 7: Display remaining scenarios
    with st.expander("Remaining delivery scenarios ⬇️"):
        for result in scenario_results:
            if result['scenario'] == best_scenario['scenario']:
                continue

            st.write(f" For the Delivery Scenario: **{result['scenario']}**")
            st.write(f"Total Consolidated Shipment Cost: **€{result['total_consolidated_cost']:,.2f}**")
            st.write(f"No of shipments: {result['num_shipments']:,}")
            st.write(f"Avg Pallets: {result['avg_pallets']}")
            st.write(50 * '-')

    charts = create_heatmap_and_bar_charts(consolidated_data, aggregated_data, parameters['start_date'], parameters['end_date'])

    years_in_range = set(pd.date_range(parameters['start_date'], parameters['end_date']).year)

    with st.expander("Heatmap Analysis Charts(Before & After Consolidation)"):
        for year in [2024]:
            if year in years_in_range:
                chart_original, chart_consolidated, bar_comparison = charts[year]

                # Display heatmaps for the current year
                st.write(f"**Visualisation using Heatmaps (Before & After Consolidation):**")
                st.components.v1.html(chart_original.render_embed(), height=216, width=1000)
                st.components.v1.html(chart_consolidated.render_embed(), height=216, width=1000)