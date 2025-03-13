import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
from itertools import product
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Calendar, Page
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar
from bokeh.palettes import RdYlGn11
from bokeh.transform import linear_cmap

global group_field
global group_method
import openai

#shipment_window_range=(1, 10)
#total_shipment_capacity= 36

import openai

def rephrase_text(api_key, text):
    """
    Sends the given text to OpenAI's ChatCompletion API for rephrasing.

    Parameters:
        api_key (str): OpenAI API key.
        text (str): The text to be rephrased.

    Returns:
        str: The rephrased text.
    """
    # Set the API key
    openai.api_key = api_key

    instructions = "Rephrase the given text in a unique and varied way while maintaining clarity, professionalism, and conciseness. Ensure the response is distinct from previous attempts."

    try:
        # Call OpenAI's API for rephrasing

        response = openai.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for optimal performance
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text}
            ],
            max_tokens=500,  # Adjust based on your text size
            temperature=1.2 # Balance creativity and accuracy
        )

        # Extract and return the rephrased text
        result = response.choices[0].message.content
        return result

    except openai.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {str(e)}"

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"



def load_data():
    rate_card_ambient = pd.read_excel('Complete Input.xlsx', sheet_name='AMBIENT')
    rate_card_ambcontrol = pd.read_excel('Complete Input.xlsx', sheet_name='AMBCONTROL')
    return rate_card_ambient, rate_card_ambcontrol

rate_card_ambient, rate_card_ambcontrol = load_data()

def get_filtered_data(parameters, df):

    global group_field
    global group_method

    group_method = parameters['group_method']
    group_field = 'SHORT_POSTCODE' if group_method == 'Post Code Level' else 'NAME'

    # Month selection
    start_date= parameters['start_date']
    end_date= parameters['end_date']

    # Filter data based on selected date range
    df = df[(df['SHIPPED_DATE'] >= start_date) & (df['SHIPPED_DATE'] <= end_date)]
    print("only date filter", df.shape) ### checkk

    # Add checkbox and conditional dropdown for selecting post codes or customers
  
    if group_method == 'Post Code Level':
        all_postcodes = parameters['all_post_code']
        
        if not all_postcodes:
            selected_postcodes = parameters['selected_postcodes']
            selected_postcodes= [z.strip('') for z in selected_postcodes ]
    else:  # Customer Level
        all_customers = parameters['all_customers']
        if not all_customers:
            selected_customers = parameters['selected_customers']
            selected_customers= [c.strip('') for c in selected_customers]
    # Filter the dataframe based on the selection
    if group_method == 'Post Code Level' and not all_postcodes:
        if selected_postcodes:  # Only filter if some postcodes are selected
            df = df[df['SHORT_POSTCODE'].str.strip('').isin(selected_postcodes)]
        else:
            return pd.DataFrame()
        
    elif group_method == 'Customer Level' and not all_customers:
        if selected_customers:  # Only filter if some customers are selected
            df = df[df['NAME'].str.strip('').isin(selected_customers)]
        else :
            return pd.DataFrame()
    
    # if len(df)>0:
    #     st.write("minimum date: ", df['SHIPPED_DATE'].min(), "maximum date: ", df['SHIPPED_DATE'].max())

    return df

# Helper functions
def calculate_priority(shipped_date, current_date, shipment_window):
    days_left = (shipped_date - current_date).days
    if 0 <= days_left <= shipment_window:
        return days_left
    return np.nan

def get_shipment_cost(prod_type, short_postcode, total_pallets):
    if prod_type == 'AMBIENT':
        rate_card = rate_card_ambient
    elif prod_type == 'AMBCONTROL':
        rate_card = rate_card_ambcontrol
    else:
        return np.nan

    row = rate_card[rate_card['SHORT_POSTCODE'] == short_postcode]
    
    if row.empty:
        return np.nan

    cost_per_pallet = row.get(total_pallets, np.nan).values[0]

    if pd.isna(cost_per_pallet):
        return np.nan

    shipment_cost = round(cost_per_pallet * total_pallets, 1)
    return shipment_cost

def get_baseline_cost(prod_type, short_postcode, pallets):
    total_cost = 0
    for pallet in pallets:
        cost = get_shipment_cost(prod_type, short_postcode, pallet)
        if pd.isna(cost):
            return np.nan
        total_cost += cost
    return round(total_cost, 1)

def best_fit_decreasing(items, capacity):
    items = sorted(items, key=lambda x: x['Total Pallets'], reverse=True)
    shipments = []

    for item in items:
        best_shipment = None
        min_space = capacity + 1

        for shipment in shipments:
            current_load = sum(order['Total Pallets'] for order in shipment)
            new_load = current_load + item['Total Pallets']
            
            if new_load <= capacity:
                space_left = capacity - current_load
            else:
                continue  # Skip this shipment if adding the item would exceed capacity
            
            if item['Total Pallets'] <= space_left < min_space:
                best_shipment = shipment
                min_space = space_left

        if best_shipment:
            best_shipment.append(item)
        else:
            shipments.append([item])

    return shipments

def process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity):
    total_pallets = sum(order['Total Pallets'] for order in shipment)
    utilization = (total_pallets / capacity) * 100

    prod_type = shipment[0]['PROD TYPE']
    short_postcode = shipment[0]['SHORT_POSTCODE']
    shipment_cost = get_shipment_cost(prod_type, short_postcode, total_pallets)

    pallets = [order['Total Pallets'] for order in shipment]
    baseline_cost = get_baseline_cost(prod_type, short_postcode, pallets)

    shipment_info = {
        'Date': current_date,
        'Orders': [order['ORDER_ID'] for order in shipment],
        'Total Pallets': total_pallets,
        'Capacity': capacity,
        'Utilization %': round(utilization, 1),
        'Order Count': len(shipment),
        'Pallets': pallets,
        'PROD TYPE': prod_type,
        'GROUP': shipment[0]['GROUP'],
        'Shipment Cost': shipment_cost,
        'Baseline Cost': baseline_cost,
        'SHORT_POSTCODE': short_postcode,
        'Load Type': 'Full' if total_pallets > 26 else 'Partial'
    }

    if group_method == 'NAME':
        shipment_info['NAME'] = shipment[0]['NAME']

    consolidated_shipments.append(shipment_info)
    
    for order in shipment:
        allocation_matrix.loc[order['ORDER_ID'], current_date] = 1
        working_df.drop(working_df[working_df['ORDER_ID'] == order['ORDER_ID']].index, inplace=True)

def consolidate_shipments(df, high_priority_limit, utilization_threshold, shipment_window, date_range, progress_callback, capacity):
    consolidated_shipments = []
    allocation_matrix = pd.DataFrame(0, index=df['ORDER_ID'], columns=date_range)
    
    working_df = df.copy()
    
    for current_date in date_range:
        working_df.loc[:, 'Priority'] = working_df['SHIPPED_DATE'].apply(lambda x: calculate_priority(x, current_date, shipment_window))

        if (working_df['Priority'] == 0).any():
            eligible_orders = working_df[working_df['Priority'].notnull()].sort_values('Priority')
            high_priority_orders = eligible_orders[eligible_orders['Priority'] <= high_priority_limit].to_dict('records')
            low_priority_orders = eligible_orders[eligible_orders['Priority'] > high_priority_limit].to_dict('records')
            
            if high_priority_orders or low_priority_orders:
                # Process high priority orders first
                high_priority_shipments = best_fit_decreasing(high_priority_orders, capacity)
                
                # Try to fill high priority shipments with low priority orders
                for shipment in high_priority_shipments:
                    current_load = sum(order['Total Pallets'] for order in shipment)
                    space_left = capacity - current_load  # Use the variable capacity
                    
                    if space_left > 0:
                        low_priority_orders.sort(key=lambda x: x['Total Pallets'], reverse=True)
                        for low_priority_order in low_priority_orders[:]:
                            if low_priority_order['Total Pallets'] <= space_left:
                                shipment.append(low_priority_order)
                                space_left -= low_priority_order['Total Pallets']
                                low_priority_orders.remove(low_priority_order)
                            if space_left == 0:
                                break
                
                # Process remaining low priority orders
                low_priority_shipments = best_fit_decreasing(low_priority_orders, capacity)
                
                # Process all shipments
                all_shipments = high_priority_shipments + low_priority_shipments
                for shipment in all_shipments:
                    total_pallets = sum(order['Total Pallets'] for order in shipment)
                    utilization = (total_pallets / capacity) * 100
                    
                    # Always process shipments with high priority orders, apply threshold only to pure low priority shipments
                    if any(order['Priority'] <= high_priority_limit for order in shipment) or utilization >= utilization_threshold:
                        process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity)
        
        progress_callback()
    
    return consolidated_shipments, allocation_matrix

def calculate_metrics(all_consolidated_shipments, df):
    total_orders = sum(len(shipment['Orders']) for shipment in all_consolidated_shipments)
    total_shipments = len(all_consolidated_shipments)
    total_pallets = sum(shipment['Total Pallets'] for shipment in all_consolidated_shipments)
    total_utilization = sum(shipment['Utilization %'] for shipment in all_consolidated_shipments)
    average_utilization = total_utilization / total_shipments if total_shipments > 0 else 0
    total_shipment_cost = sum(shipment['Shipment Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Shipment Cost']))
    total_baseline_cost = sum(shipment['Baseline Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Baseline Cost']))
    cost_savings = total_baseline_cost - total_shipment_cost
    percent_savings = (cost_savings / total_baseline_cost) * 100 if total_baseline_cost > 0 else 0

    # Calculate CO2 Emission
    total_distance = 0
    sum_dist = 0
    for shipment in all_consolidated_shipments:
        order_ids = shipment['Orders']
        avg_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].mean()
        sum_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].sum()
        total_distance += avg_distance
        sum_dist += sum_distance
    co2_emission = (sum_dist - total_distance) * 2  # Multiply by 2 


    metrics = {
        'Total Orders': total_orders,
        'Total Shipments': total_shipments,
        'Total Pallets': total_pallets,
        'Average Utilization': average_utilization,
        'Total Shipment Cost': total_shipment_cost,
        'Total Baseline Cost': total_baseline_cost,
        'Cost Savings': round(cost_savings,1),
        'Percent Savings': percent_savings,
        'CO2 Emission': co2_emission
    }

    return metrics

def create_utilization_chart(all_consolidated_shipments):
    #print(all_consolidated_shipments)  #### checkkkk
    utilization_bins = {f"{i}-{i+5}%": 0 for i in range(0, 100, 5)}
    for shipment in all_consolidated_shipments:
        utilization = shipment['Utilization %']
        bin_index = min(int(utilization // 5) * 5, 95)  # Cap at 95-100% bin
        bin_key = f"{bin_index}-{bin_index+5}%"
        utilization_bins[bin_key] += 1

    total_shipments = len(all_consolidated_shipments)
    utilization_distribution = {bin: (count / total_shipments) * 100 for bin, count in utilization_bins.items()}

    fig = go.Figure(data=[go.Bar(x=list(utilization_distribution.keys()), y=list(utilization_distribution.values()), marker_color='#1f77b4')])
    fig.update_layout(
        title={'text': 'Utilization Distribution', 'font': {'size': 22}},
        xaxis_title='Utilization Range',
        yaxis_title='Percentage of Shipments', 
        width=1000, 
        height=500
        )

    return fig


def create_consolidated_shipments_calendar(consolidated_df):
    # Group by Date and calculate both Shipments Count and Total Orders
    df_consolidated = consolidated_df.groupby('Date').agg({
        'Orders': ['count', lambda x: sum(len(orders) for orders in x)]
    }).reset_index()
    df_consolidated.columns = ['Date', 'Shipments Count', 'Orders Count']
    
    # Split data by year
    df_2023 = df_consolidated[df_consolidated['Date'].dt.year == 2023]
    df_2024 = df_consolidated[df_consolidated['Date'].dt.year == 2024]
    df_2025 = df_consolidated[df_consolidated['Date'].dt.year == 2025]
    
    calendar_data_2023 = df_2023[['Date', 'Shipments Count', 'Orders Count']].values.tolist()
    calendar_data_2024 = df_2024[['Date', 'Shipments Count', 'Orders Count']].values.tolist()
    calendar_data_2025 = df_2025[['Date', 'Shipments Count', 'Orders Count']].values.tolist()

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
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Orders and Shipments After Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[2] for item in data) if data else 0,
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
                                   '<br/>Orders: ' + p.data[2] +
                                   '<br/>Shipments: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)


    return calendar_2023, calendar_2024 , calendar_2025

def create_original_orders_calendar(original_df):
    df_original = original_df.groupby('SHIPPED_DATE').size().reset_index(name='Orders Shipped')
    
    # Split data by year
    df_2023 = df_original[df_original['SHIPPED_DATE'].dt.year == 2023]
    df_2024 = df_original[df_original['SHIPPED_DATE'].dt.year == 2024]
    df_2025 = df_original[df_original['SHIPPED_DATE'].dt.year == 2025]
    
    calendar_data_2023 = df_2023[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()
    calendar_data_2024 = df_2024[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()
    calendar_data_2025 = df_2025[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()

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
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Orders Shipped Before Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[1] for item in data) if data else 0,
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
                            return 'Date: ' + day + '/' + month + '/' + year + '<br/>Orders: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)

    return calendar_2023, calendar_2024 , calendar_2025

def create_heatmap_and_bar_charts(consolidated_df, original_df, start_date, end_date):
    # Create calendar charts (existing code)
    chart_original_2023, chart_original_2024  , chart_original_2025 = create_original_orders_calendar(original_df)
    chart_consolidated_2023, chart_consolidated_2024 , chart_consolidated_2025 = create_consolidated_shipments_calendar(consolidated_df)
    
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
    bar_charts_2025 = create_bar_charts(original_df, consolidated_df, 2025)
    
    return {
        2023: (chart_original_2023, chart_consolidated_2023, bar_charts_2023),
        2024: (chart_original_2024, chart_consolidated_2024, bar_charts_2024),
        2025: (chart_original_2025, chart_consolidated_2025, bar_charts_2025)
    }



def analyze_consolidation_distribution(all_consolidated_shipments, df):
    distribution = {}
    for shipment in all_consolidated_shipments:
        consolidation_date = shipment['Date']
        for order_id in shipment['Orders']:
            shipped_date = df.loc[df['ORDER_ID'] == order_id, 'SHIPPED_DATE'].iloc[0]
            days_difference = (shipped_date - consolidation_date).days
            if days_difference not in distribution:
                distribution[days_difference] = 0
            distribution[days_difference] += 1
    
    total_orders = sum(distribution.values())
    distribution_percentage = {k: round((v / total_orders) * 100, 1) for k, v in distribution.items()}
    return distribution, distribution_percentage



def create_metric_box(label, value, color_start="#1f77b4", color_end="#0053a4"):
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
        padding: 5px;
        border-radius: 15px;
        margin: 0px 0 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        border: 2px solid #ffffff;
        position: relative;
        overflow: hidden;
    ">
        <p style="color: white; margin: 0; font-size: 15px; font-weight: 600; position: relative; z-index: 1;">{label}</p>
        <p style="color: white; font-size: 17px; font-weight: 600; margin: 0px 0 0 0; position: relative; z-index: 1;">{value}</p>
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: transparent;
            pointer-events: none;
        "></div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)


st.markdown("""
<style>
/* Style for regular buttons */
div.stButton > button:first-child {
    background-color: #c06238;
    color: white;
}
div.stButton > button:hover {
    background-color: #0053a4;
    color: white;
}

/* Style for download buttons */
div.stDownloadButton > button:first-child {
    background-color: #c06238;
    color: white;
}
div.stDownloadButton > button:hover {
    background-color: #0053a4;
    color: white;
}
</style>
""", unsafe_allow_html=True)



def run_cost_optimization_simulation(parameters , api_key , total_capacity,shipment_window , customers , postcodes):

    start_date= parameters['start_date']
    end_date= parameters['end_date']

    df = pd.read_excel('Complete Input.xlsx', sheet_name='Sheet1')
    df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)
    df= get_filtered_data(parameters, df)
    
    if len(df)==0:
        st.write("No data is available for the selected parameters., please try agian !")
        return None

    # Prepare data for simulation
    df['GROUP'] = df[group_field]
    grouped = df.groupby(['PROD TYPE', 'GROUP'])
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Generate all combinations of parameters
    shipment_windows = range(shipment_window[0], shipment_window[1] + 1)
    utilization_threshold = 95
    
    total_groups = len(grouped)
    total_days = len(date_range)
    total_iterations = len(shipment_windows) * total_groups * total_days

    # Initialize variables to store best results
    best_metrics = None
    best_consolidated_shipments = None
    best_params = None
    
    all_results = []
    iteration_counter = 0
    
    # Run simulation for each combination
    for shipment_window in shipment_windows:
        high_priority_limit = 0
        all_consolidated_shipments = []
    
        for (prod_type, group), group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, high_priority_limit, utilization_threshold, 
                shipment_window, date_range, lambda: None, total_capacity
            )
            all_consolidated_shipments.extend(consolidated_shipments)
            
            iteration_counter += total_days
            progress_percentage = int((iteration_counter / total_iterations) * 100)
            print(f"Progress: {progress_percentage}%")
    
        # Calculate metrics for this combination
        metrics = calculate_metrics(all_consolidated_shipments, df)
        
        # Analyze consolidation distribution
        distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)
        
        result = {
            'Shipment Window': shipment_window,
            'Total Orders': metrics['Total Orders'],
            'Total Shipments': metrics['Total Shipments'],
            'Total Shipment Cost': round(metrics['Total Shipment Cost'], 1),
            'Total Baseline Cost': round(metrics['Total Baseline Cost'], 1),
            'Cost Savings': metrics['Cost Savings'],
            'Percent Savings': round(metrics['Percent Savings'], 1),
            'Average Utilization': round(metrics['Average Utilization'], 1),
            'CO2 Emission': round(metrics['CO2 Emission'], 1)
        }
        
        # Add columns for days relative to shipped date
        for i in range(shipment_window + 1):
            column_name = f'orders%_shipping_{i}day_early'
            result[column_name] = distribution_percentage.get(i, 0)
        
        all_results.append(result)
    
        # Update best results if current combination is better
        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
            best_metrics = metrics
            best_consolidated_shipments = all_consolidated_shipments
            best_params = (shipment_window, high_priority_limit, utilization_threshold)
    
    # end_time = time.time()
    # time_taken = end_time - start_time
    # time_taken_minutes = int(time_taken // 60)
    # time_taken_seconds = int(time_taken % 60)
    # st.write(f"Time taken: {time_taken_minutes} minutes {time_taken_seconds} seconds")

        # Display best results
    #st.markdown("<h2 style='font-size:26px;'>Best Simulation Results</h2>", unsafe_allow_html=True)
    #st.write(f"The optimal Shipment Window for the selected data is: {best_params[0]}")


    with st.expander("VIEW APPROACH OF COST CONSOLIDATION", expanded=False):
        summary_text = (
            f"Optimizing outbound deliveries and identifying cost-saving opportunities involve analyzing various factors "
            f"such as order patterns, delivery routes, shipping costs, and consolidation opportunities.\n\n"

            f"On analyzing the data, I can provide some estimates of cost savings on the historical data if we were to "
            f"group orders to consolidate deliveries.\n\n"

            "**APPROACH TAKEN**\n\n"
            f"To consolidate the deliveries, A heuristic approach was used, and the methodology is as follows:\n\n"

            f"**Group Shipments**: Orders are consolidated within a shipment window to reduce transportation costs while "
            f"maintaining timely deliveries. A shipment window represents the number of days prior to the current delivery "
            f"that the order could be potentially shipped, thus representing an opportunity to group it with earlier deliveries.\n\n"

            f"**Iterate Over Shipment Windows**: The model systematically evaluates all possible shipment windows, testing "
            f"different configurations to identify the most effective scheduling approach.\n\n"

            f"**Performance Metric Calculation**: Key performance metrics are assessed for each shipment window, including:\n"
            f"- **Cost savings**\n"
            f"- **Utilization rate**\n"
            f"- **CO2 emission reduction**\n\n"

            f"**Comparison and Selection**: After evaluating all configurations, the shipment window that maximizes cost savings "
            f"while maintaining operational efficiency is identified, and results are displayed as per the best parameter.\n\n"

            f"This method allows us to optimize logistics operations dynamically, ensuring that both financial and environmental "
            f"factors are balanced effectively."
        )
        rephrased_text = rephrase_text(api_key, summary_text)
        st.write(rephrased_text)



    # Display the summary sentence before calculations start
    #col1, col2, col3 = st.columns(3)

    calc_shipment_window = best_params[0]

    calc_utilization_threshold = 95

    if True:
        # start_time = time.time()

        with st.spinner("Calculating..."):
            # Prepare data for calculation
            df['GROUP'] = df[group_field]
            grouped = df.groupby(['PROD TYPE', 'GROUP'])
            date_range = pd.date_range(start=start_date, end=end_date)

            calc_high_priority_limit = 0
            all_consolidated_shipments = []
            all_allocation_matrices = []

            # Run calculation


            for i, ((prod_type, group), group_df) in enumerate(grouped):
                consolidated_shipments, allocation_matrix = consolidate_shipments(
                    group_df, calc_high_priority_limit, calc_utilization_threshold,
                    calc_shipment_window, date_range, lambda: None, total_capacity
                )
                all_consolidated_shipments.extend(consolidated_shipments)
                all_allocation_matrices.append(allocation_matrix)

            selected_postcodes = ", ".join(postcodes) if postcodes else "All Postcodes"
            selected_customers = ", ".join(customers) if customers else "All Customers"

            metrics = calculate_metrics(all_consolidated_shipments, df)


        with st.expander("IDENTIFIED COST SAVINGS AND KEY PERFORMANCE INDICATORS(KPIs)", expanded=False):
            main_text = (
                    f"Through extensive analysis, the **OPTIMAL SHIPMENT WINDOW** was determined to be **{best_params[0]}**, "
                    f"with a PALLET SIZE of **{total_capacity}** for **postcodes**: {selected_postcodes} and **customers**: {selected_customers}."
                    f"These optimizations resulted in SIGNIFICANT EFFICIENCY IMPROVEMENTS:\n\n"

                    f"**SHIPMENT WINDOW**: The most effective shipment window was identified as **{best_params[0]} DAYS**.\n\n"

                    f"**COST SAVINGS**: A reduction of **£{metrics['Cost Savings']:,.1f}**, equating to an **£{metrics['Percent Savings']:.1f}%** decrease in overall transportation costs.\n\n"

                    f"**ORDER & SHIPMENT SUMMARY**:\n"
                    f"- **TOTAL ORDERS PROCESSED**: {metrics['Total Orders']:,} \n"
                    f"- **TOTAL SHIPMENTS MADE**: {metrics['Total Shipments']:,}\n\n"

                    f"**UTILIZATION EFFICIENCY**:\n"
                    f"- **AVERAGE TRUCK UTILIZATION** increased to **{metrics['Average Utilization']:.1f}%**, ensuring fewer trucks operate at low capacity.\n\n"

                    f"**ENVIRONMENTAL IMPACT**:\n"
                    f"- **CO2 EMISSIONS REDUCTION**: A decrease of **{metrics['CO2 Emission']:,.1f} Kg**, supporting sustainability efforts and reducing the carbon footprint.\n\n"

                    f"These optimizations not only lead to substantial **COST REDUCTIONS** but also enhance **OPERATIONAL SUSTAINABILITY**, "
                    f"allowing logistics operations to function more efficiently while **MINIMIZING ENVIRONMENTAL IMPACT**."
            )
            rephrase_main_text = rephrase_text(api_key, main_text)
            st.write(rephrase_main_text)


    # Save results and charts
    utilization_chart = create_utilization_chart(best_consolidated_shipments)

    # Create a dataframe with all simulation results
    results_df = pd.DataFrame(all_results)
    
    # Preprocess the data to keep only the row with max Cost Savings for each Shipment Window
    optimal_results = results_df.loc[results_df.groupby(['Shipment Window'])['Cost Savings'].idxmax()]
    
    # Create ColumnDataSource
    source = ColumnDataSource(optimal_results)

    
    # Display the Shipment Window Comparison chart
    #st.markdown("<h2 style='font-size:24px;'>Shipment Window Comparison</h2>", unsafe_allow_html=True)
    with st.expander("SHIPMENT WINDOW COMPARISION", expanded=False):

        shipment_text = (
            f"For each shipment window:\n\n"
            f"- Shipments are grouped together through the consolidation function.\n"
            f"- Key performance metrics, such as cost savings, utilization, and emissions, are calculated.\n"
            f"- The cost savings are compared across different shipment windows to identify the most efficient one.\n"
            f"- On analyzing this data , the best shipment window is observed to be  **{best_params[0]}** days."
        )
        shipment_rephrase_text = rephrase_text(api_key, shipment_text)
        st.write(shipment_rephrase_text)

        # Select the best rows for each shipment window
        best_results = results_df.loc[results_df.groupby('Shipment Window')['Percent Savings'].idxmax()]

        # Sort by Shipment Window
        best_results = best_results.sort_values('Shipment Window')

        # Create a complete range of shipment windows from 0 to 30
        all_windows = list(range(0, 31))

        # Create the subplot figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add the stacked bar chart
        fig.add_trace(
            go.Bar(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Total Shipment Cost'].values[0] if w in
                                                                                                          best_results[
                                                                                                              'Shipment Window'].values else 0
                   for w in all_windows],
                name='Total Shipment Cost',
                marker_color='#1f77b4'
            )
        )

        fig.add_trace(
            go.Bar(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Cost Savings'].values[0] if w in best_results[
                    'Shipment Window'].values else 0 for w in all_windows],
                name='Cost Savings',
                marker_color='#a9d6a9'
            )
        )

        # Add the line chart for Total Shipments on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Total Shipments'].values[0] if w in best_results[
                    'Shipment Window'].values else None for w in all_windows],
                name='Total Shipments',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8),
                hovertemplate='<b>Shipment Window</b>: %{x}<br>' +
                              '<b>Total Shipments</b>: %{y}<br>' +
                              '<b>Average Utilization</b>: %{text:.1f}%<extra></extra>',
                text=[best_results[best_results['Shipment Window'] == w]['Average Utilization'].values[0] if w in
                                                                                                             best_results[
                                                                                                                 'Shipment Window'].values else None
                      for w in all_windows],
            ),
            secondary_y=True
        )

        # Add text annotations for Percent Savings
        for w in all_windows:
            if w in best_results['Shipment Window'].values:
                row = best_results[best_results['Shipment Window'] == w].iloc[0]
                fig.add_annotation(
                    x=w,
                    y=row['Total Shipment Cost'] + row['Cost Savings'],
                    text=f"{row['Percent Savings']:.1f}%",
                    showarrow=False,
                    yanchor='bottom',
                    yshift=5,
                    font=dict(size=10)
                )

        # Update the layout
        fig.update_layout(
            barmode='stack',
            height=600,
            width=1050,
            # margin=dict(l=50, r=50, t=40, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_xaxes(title_text='Shipment Window', tickmode='linear', dtick=1, range=[-0.5, 30.5])
        fig.update_yaxes(title_text='Cost (£)', secondary_y=False)
        fig.update_yaxes(title_text='Total Shipments', secondary_y=True)

        # Show the chart
        st.plotly_chart(fig, use_container_width=False)

    # Return results
    return {
        "metrics": best_metrics,
        "results_df": results_df,
        "charts": utilization_chart,
        "params": best_params
    }

# # Calculation tab
def cost_calculation(parameters, best_params , total_capacity):

    df = pd.read_excel('Complete Input.xlsx', sheet_name='Sheet1')
    df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)

    df= get_filtered_data(parameters, df)
    start_date= parameters['start_date']
    end_date= parameters['end_date']

    #st.markdown("<h2 style='font-size:24px;'>Calculation Parameters</h2>", unsafe_allow_html=True)


    calc_shipment_window = best_params[0]

    calc_utilization_threshold = 95

    if True:
        # start_time = time.time()

        with st.spinner("Calculating..."):
            # Prepare data for calculation
            df['GROUP'] = df[group_field]
            grouped = df.groupby(['PROD TYPE', 'GROUP'])
            date_range = pd.date_range(start=start_date, end=end_date)

            calc_high_priority_limit = 0
            all_consolidated_shipments = []
            all_allocation_matrices = []

            # Run calculation
            #progress_bar = custom_progress_bar()

            for i, ((prod_type, group), group_df) in enumerate(grouped):
                consolidated_shipments, allocation_matrix = consolidate_shipments(
                    group_df, calc_high_priority_limit, calc_utilization_threshold,
                    calc_shipment_window, date_range, lambda: None, total_capacity
                )
                all_consolidated_shipments.extend(consolidated_shipments)
                all_allocation_matrices.append(allocation_matrix)
                #progress_percentage = int(((i + 1) / len(grouped)) * 100)
                #progress_bar(progress_percentage)

            # end_time = time.time()
            # time_taken = end_time - start_time
            # time_taken_minutes = int(time_taken // 60)
            # time_taken_seconds = int(time_taken % 60)
            # st.write(f"Time taken: {time_taken_minutes} minutes {time_taken_seconds} seconds")

            consolidated_df = pd.DataFrame(all_consolidated_shipments)

            st.write("View Detailed Metrics Summary")

            days_shipped_before = df['SHIPPED_DATE'].nunique()
            total_pallets_before = df['Total Pallets'].sum()
            pallets_per_day_before = total_pallets_before / days_shipped_before
            total_orders_before = len(df)
            pallets_per_shipment_before = total_pallets_before / total_orders_before  # Each order is a shipment

            # Calculate metrics for after consolidation
            days_shipped_after = consolidated_df['Date'].nunique()
            total_pallets_after = consolidated_df['Total Pallets'].sum()
            pallets_per_day_after = total_pallets_after / days_shipped_after
            total_shipments_after = len(consolidated_df)
            pallets_per_shipment_after = total_pallets_after / total_shipments_after

            # Calculate percentage changes
            days_change = ((days_shipped_after - days_shipped_before) / days_shipped_before) * 100
            pallets_per_day_change = ((pallets_per_day_after - pallets_per_day_before) / pallets_per_day_before) * 100
            pallets_per_shipment_change = ((
                                                       pallets_per_shipment_after - pallets_per_shipment_before) / pallets_per_shipment_before) * 100

            # Create three columns for before, after, and change metrics
            col1, col2, col3 = st.columns(3)

            # Style for metric display
            metric_style = """
                                <div style="
                                    background-color: #f0f2f6;
                                    padding: 0px;
                                    border-radius: 5px;
                                    margin: 5px 0;
                                ">
                                    <span style="font-weight: bold;">{label}:</span> {value}
                                </div>
                            """

            # Style for percentage changes
            change_style = """
                                <div style="
                                    background-color: #e8f0fe;
                                    padding: 0px;
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

            # Before consolidation metrics
            with col1:
                st.markdown("##### Before Consolidation")
                st.markdown(metric_style.format(
                    label="Days Shipped",
                    value=f"{days_shipped_before:,}"
                ), unsafe_allow_html=True)
                st.markdown(metric_style.format(
                    label="Pallets Shipped per Day",
                    value=f"{pallets_per_day_before:.1f}"
                ), unsafe_allow_html=True)
                st.markdown(metric_style.format(
                    label="Pallets per Shipment",
                    value=f"{pallets_per_shipment_before:.1f}"
                ), unsafe_allow_html=True)

            # After consolidation metrics
            with col2:
                st.markdown("##### After Consolidation")
                st.markdown(metric_style.format(
                    label="Days Shipped",
                    value=f"{days_shipped_after:,}"
                ), unsafe_allow_html=True)
                st.markdown(metric_style.format(
                    label="Pallets Shipped per Day",
                    value=f"{pallets_per_day_after:.1f}"
                ), unsafe_allow_html=True)
                st.markdown(metric_style.format(
                    label="Pallets per Shipment",
                    value=f"{pallets_per_shipment_after:.1f}"
                ), unsafe_allow_html=True)

            # Percentage changes
            with col3:
                st.markdown("##### Percentage Change")
                st.markdown(change_style.format(
                    label="Days Shipped",
                    value=days_change,
                    color="blue" if days_change > 0 else "green"
                ), unsafe_allow_html=True)
                st.markdown(change_style.format(
                    label="Pallets Shipped per Day",
                    value=pallets_per_day_change,
                    color="green" if pallets_per_day_change > 0 else "red"
                ), unsafe_allow_html=True)
                st.markdown(change_style.format(
                    label="Pallets per Shipment",
                    value=pallets_per_shipment_change,
                    color="green" if pallets_per_shipment_change > 0 else "red"
                ), unsafe_allow_html=True)

                st.text("")

            charts = create_heatmap_and_bar_charts(consolidated_df, df, start_date, end_date)

            # years_in_range = set(pd.date_range(start_date, end_date).year)

            with st.expander("Heatmap Analysis Charts(Before & After Consolidation)"):
                years_in_range = set(pd.date_range(start_date, end_date).year)

                for year in [2023,2024,2025]:
                    if year in years_in_range :
                        chart_original, chart_consolidated, bar_comparison = charts[year]

                        # Display heatmaps for the current year
                        st.write(f"**Heatmaps for the year {year} (Before & After Consolidation):**")
                        st.components.v1.html(chart_original.render_embed(), height=216, width=1000)
                        st.components.v1.html(chart_consolidated.render_embed(), height=216, width=1000)

                # After the loop, you can add the interpretation section just once
                st.write("""
                            **Heatmap Interpretation:**

                            - **Dark Green Areas**: Indicate high shipment concentration on specific dates, showcasing peak activity where most orders are processed.
                            - **Lighter Green Areas**: Represent fewer or no shipments, highlighting potential inefficiencies in the initial shipment strategy before optimization.

                            **Before Consolidation:**

                            - Shipments were frequent but scattered across multiple days without strategic grouping.
                            - Increased costs due to multiple underutilized shipments.
                            - Truck utilization remained suboptimal, leading to excess operational expenses.

                            **After Consolidation:**

                            - Orders were intelligently grouped into fewer shipments, reducing the total number of trips while maintaining service levels.
                            - Optimized cost savings through better utilization and fewer underfilled shipments.
                            - Enhanced planning efficiency, enabling better decision-making for future shipment scheduling.
                            """)


