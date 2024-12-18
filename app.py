import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Initialize the Dash app with title and favicon
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Cross-Sell and Retention Tool',
    update_title=None  # Removes "Updating..." from browser tab
)

# Set the favicon (browser tab icon)
app._favicon = "assets/favicon.ico"

# Custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .nav-link.active {
                box-shadow: 0 4px 8px rgba(128, 0, 0, 0.3) !important;
                background-color: #fff !important;
                color: maroon !important;
                font-weight: bold;
                border-bottom: 3px solid maroon !important;
            }
            .nav-link {
                color: #666 !important;
                font-size: 1.1rem;
                padding: 1rem 2rem !important;
                margin-right: 0.5rem;
                border: 1px solid #ddd !important;
                border-radius: 8px 8px 0 0 !important;
            }
            .nav-tabs {
                border-bottom: 2px solid #ddd !important;
                margin-bottom: 2rem !important;
            }
            .info-box {
                background: #f8f8f8;
                border-radius: 10px;
                margin: 1.5rem 0;
                padding: 1.5rem 2rem;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border: 1px solid #444444;
                transition: all 0.3s ease;
            }
            .info-box:hover {
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
            }
            .info-text {
                color: rgba(128, 0, 0, 0.9);
                font-size: 1.15rem;
                font-weight: 500;
                line-height: 1.6;
                margin: 0;
            }
            .filter-text {
                color: rgba(128, 0, 0, 0.7);
                font-size: 0.95rem;
                font-style: italic;
                margin-top: 0.8rem;
                line-height: 1.4;
            }
            .page-container {
                padding: 0 10%;
                max-width: 100%;
                margin: 0 auto;
            }
            .filter-section {
                padding: 1.5rem;
                background: #fff;
                border-bottom: 1px solid #ddd;
                margin-bottom: 2rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load data from JSON
with open('data/sample_data.json', 'r') as f:
    DATA = json.load(f)

def generate_realistic_data(filters):
    products = DATA['enterprise_cross_sell']['products']
    size = len(products)
    penetration = np.zeros((size, size))
    base_rates = DATA['enterprise_cross_sell']['penetration_matrix']['base']
    multipliers = DATA['enterprise_cross_sell']['penetration_matrix']['multipliers']
    
    # Set base penetration rates
    for i in range(size):
        for j in range(size):
            if i == j:
                penetration[i][j] = 1.0
            elif i < 7 and j < 7:  # Personal Banking
                penetration[i][j] = np.random.uniform(
                    base_rates['personal_banking']['min'],
                    base_rates['personal_banking']['max']
                )
            elif i >= 7 and j >= 7:  # Business Banking
                penetration[i][j] = np.random.uniform(
                    base_rates['business_banking']['min'],
                    base_rates['business_banking']['max']
                )
            else:  # Cross-segment
                penetration[i][j] = np.random.uniform(
                    base_rates['cross_segment']['min'],
                    base_rates['cross_segment']['max']
                )
    
    # Apply filters
    customer_types = filters.get('customer_type', ['ALL'])
    if 'ALL' not in customer_types:
        # Calculate average multiplier for selected customer types
        type_mult = np.mean([multipliers['customer_type'][ct] for ct in customer_types])
        penetration = penetration * type_mult
    
    employment_statuses = filters.get('employment_status', ['ALL'])
    if 'ALL' not in employment_statuses:
        # Calculate average multipliers for personal and business banking
        emp_personal_mult = np.mean([multipliers['employment_status'][es]['personal'] 
                                   for es in employment_statuses])
        emp_business_mult = np.mean([multipliers['employment_status'][es]['business'] 
                                   for es in employment_statuses])
        # Apply to respective sections
        penetration[:7, :] = penetration[:7, :] * emp_personal_mult
        penetration[7:, :] = penetration[7:, :] * emp_business_mult
    
    segments = filters.get('segment', ['ALL'])
    if 'ALL' not in segments:
        # Calculate average multiplier for selected segments
        segment_mult = np.mean([multipliers['segment'][seg] for seg in segments])
        penetration = penetration * segment_mult
    
    return np.minimum(penetration, 1.0)

def create_heatmap(data, anchor_product='ALL'):
    products = DATA['enterprise_cross_sell']['products']
    
    # If anchor product is selected, show only that row
    if anchor_product != 'ALL':
        anchor_idx = products.index(anchor_product)
        data_row = data[anchor_idx:anchor_idx+1, :]
        y_labels = [anchor_product]
    else:
        data_row = data
        y_labels = products

    fig = go.Figure(data=go.Heatmap(
        z=data_row,
        x=products,
        y=y_labels,
        hoverongaps=False,
        colorscale=[[0, 'white'], [1, 'maroon']],
        showscale=True,
        colorbar=dict(
            title='%',
            tickformat=',.0%',
            titleside='right'
        ),
        hovertemplate='Anchor Product: %{y}<br>Cross-sell Product: %{x}<br>Penetration: %{z:.1%}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            side='bottom',
            showgrid=False
        ),
        yaxis=dict(
            tickangle=0,
            showgrid=False
        ),
        height=700 if anchor_product == 'ALL' else 250,  # Adjust height based on selection
        autosize=True,
        margin=dict(t=20, b=100, l=150, r=50),
        plot_bgcolor='white'
    )

    return fig

def get_filter_text(anchor_product, customer_types, employment_status, segments):
    filters = []
    if anchor_product != 'ALL':
        filters.append(f"Anchor Product: {anchor_product}")
    if 'ALL' not in customer_types:
        types = ['First Product' if t == 'NTB' else 'Overall' for t in customer_types]
        filters.append(f"Customer Types: {', '.join(types)}")
    if 'ALL' not in employment_status:
        filters.append(f"Employment: {', '.join(s.title().replace('_', ' ') for s in employment_status)}")
    if 'ALL' not in segments:
        filters.append(f"Segments: {', '.join(s.title() for s in segments)}")
    
    return " | ".join(filters) if filters else "Showing all products and segments"

def generate_time_series_data(filters):
    weeks = 52
    # Generate dates in chronological order (Jan to Nov)
    dates = [(datetime.now() - timedelta(weeks=weeks-1-x)).strftime('%Y-%m-%d') for x in range(weeks)]
    
    # Get base parameters
    base_params = DATA['enterprise_cross_sell']['active_customers']['base_weekly']['ALL']
    customer_types = filters.get('customer_type', ['ALL'])
    
    # Start with higher value and decrease
    start_value = 250000  # Starting value around 250k
    end_value = 96000    # Target end value around 96k
    
    # Generate exponential decay
    decay_rate = -np.log(end_value / start_value) / weeks
    base = start_value * np.exp(-decay_rate * np.arange(weeks))
    noise = np.random.normal(0, base_params['std'], weeks)
    customers = base + noise
    
    # Apply filters
    if customer_types != ['ALL']:
        if 'NTB' in customer_types:
            customers = customers * 0.3  # 30% of total base
        elif 'ETB' in customer_types:
            customers = customers * 0.7  # 70% of total base
    
    # Apply segment multipliers
    segments = filters.get('segment', ['ALL'])
    if segments != ['ALL']:
        segment_mult = sum(DATA['enterprise_cross_sell']['active_customers']['segment_multipliers'][seg] 
                         for seg in segments if seg != 'ALL')
        customers = customers * segment_mult
    
    # Apply employment multipliers
    employment_status = filters.get('employment_status', ['ALL'])
    if employment_status != ['ALL']:
        emp_mult = sum(DATA['enterprise_cross_sell']['active_customers']['employment_multipliers'][emp] 
                      for emp in employment_status if emp != 'ALL')
        customers = customers * emp_mult
    
    df = pd.DataFrame({
        'Date': dates,
        'Active Customers': customers.astype(int)
    })
    
    return df

def generate_commentary(df, filters):
    # Get the earliest (Jan 2024) and latest (Nov 2024) values
    jan_2024 = df['Active Customers'].iloc[0]  # January 2024
    nov_2024 = df['Active Customers'].iloc[-1]  # November 2024
    
    # Calculate percentage decrease
    change = ((jan_2024 - nov_2024) / jan_2024 * 100)
    
    # Build commentary
    commentary = f"Active customer base has decreased by {abs(change):.1f}% from January to November 2024"
    
    # Add customer type context if filtered
    if filters['customer_type'] != ['ALL']:
        customer_type_text = "first-time customers" if filters['customer_type'][0] == 'NTB' else "existing customers"
        commentary += f" for {customer_type_text}"
    
    # Add segment context if filtered
    if filters['segment'] != ['ALL']:
        segment_names = [s.title() for s in filters['segment']]
        commentary += f" in the {', '.join(segment_names)} segment"
    
    # Add employment status if filtered
    if filters['employment_status'] != ['ALL']:
        emp_status = filters['employment_status'][0].replace('_', ' ').title()
        commentary += f" ({emp_status})"
    
    # Add current base number (using November 2024 value)
    commentary += f". Current active customer base: {nov_2024:,.0f}"
    
    return commentary

def get_retention_filter_text(filters):
    text_parts = []
    
    if filters['anchor_product'] != 'ALL':
        text_parts.append(f"Anchor Product: {filters['anchor_product']}")
    
    if filters['customer_type'] != 'ALL':
        customer_type_text = "First-time customers" if filters['customer_type'] == 'NTB' else "Existing customers"
        text_parts.append(customer_type_text)
    
    if filters['employment_status'] != 'ALL':
        emp_status = filters['employment_status'].replace('_', ' ').title()
        text_parts.append(emp_status)
    
    if filters['segment'] != 'ALL':
        text_parts.append(filters['segment'].title())
    
    return " | ".join(text_parts) if text_parts else "All Customers"

def generate_inactive_funnel_data(filters):
    retention_data = DATA['retention']['inactive_funnel']
    base_data = retention_data['base_distribution']
    
    # Initialize funnel data
    funnel_data = []
    total_customers = 0
    
    # Get base value based on anchor product
    if filters['anchor_product'] != 'ALL':
        product_data = retention_data['product_factors'][filters['anchor_product']]
        base_value = product_data['base']
    else:
        base_value = base_data['0_month']['base']
    
    # Apply multipliers
    if filters['customer_type'] != 'ALL':
        base_value *= retention_data['customer_type_multipliers'][filters['customer_type']]['factor']
        
    if filters['segment'] != 'ALL':
        base_value *= retention_data['segment_multipliers'][filters['segment']]['factor']
        
    if filters['employment_status'] != 'ALL':
        base_value *= retention_data['employment_multipliers'][filters['employment_status']]['factor']
    
    # Calculate values for each month
    for month, data in base_data.items():
        value = base_value * data['proportion']
        month_label = f"{month.split('_')[0]} Month{'s' if month != '0_month' else ''} Inactive"
        funnel_data.append({
            'stage': month_label,
            'value': int(value)
        })
        total_customers += int(value)
    
    return funnel_data, total_customers

def create_inactive_funnel(funnel_data, total_customers):
    # Create funnel chart using Plotly's Figure Factory
    fig = go.Figure()
    
    # Add funnel trace
    fig.add_trace(go.Funnel(
        name='Inactive Customers',
        y=[d['stage'] for d in funnel_data],
        x=[d['value'] for d in funnel_data],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.85,
        marker={
            "color": ["maroon", "#B22222", "#CD5C5C", "#E68080", "#F0A0A0", "#F8C0C0", "#FFE0E0"],
            "line": {"width": [1, 1, 1, 1, 1, 1, 1], "color": ["white", "white", "white", "white", "white", "white", "white"]}
        },
        hovertemplate="<b>%{y}</b><br>" +
                     "Customers: %{x:,.0f}<br>" +
                     "% of Total: %{percentInitial:.1%}<extra></extra>"
    ))
    
    # Calculate dynamic height based on number of stages
    height = max(400, min(len(funnel_data) * 100, 700))
    
    # Calculate dynamic width based on largest value
    max_value = max(d['value'] for d in funnel_data)
    width = max(600, min(max_value / 100, 1000))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Inactive Customer Distribution<br><span style="font-size: 14px">Total Inactive Customers: {total_customers:,}</span>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font_size=12,
        showlegend=False,
        margin=dict(t=100, b=20, l=120, r=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=height,
        width=width,
        funnelmode="stack",
        funnelgap=0.2
    )
    
    # Update axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig

# Create layout with tabs
app.layout = html.Div([
    html.Div([  # Add wrapper div for page margins
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(label="Enterprise Cross-Sell", tab_id="cross-sell", children=[
                    # Filters Row
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Anchor Product"),
                                dcc.Dropdown(
                                    id='anchor-product',
                                    options=[{'label': 'All Products', 'value': 'ALL'}] + [
                                        {'label': product, 'value': product}
                                        for product in DATA['enterprise_cross_sell']['products']
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Customer Type"),
                                dcc.Dropdown(
                                    id='customer-type',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'First Product', 'value': 'NTB'},
                                        {'label': 'Overall', 'value': 'ETB'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Employment Status"),
                                dcc.Dropdown(
                                    id='employment-status',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'Salaried', 'value': 'SALARIED'},
                                        {'label': 'Self Employed', 'value': 'SELF_EMPLOYED'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Segment"),
                                dcc.Dropdown(
                                    id='segment',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'Mass', 'value': 'MASS'},
                                        {'label': 'Elite', 'value': 'ELITE'},
                                        {'label': 'Select', 'value': 'SELECT'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3)
                        ])
                    ], className="filter-section"),
                    
                    # Description Box with dynamic filters
                    html.Div([
                        html.P(
                            "Cross-sell penetration by anchor product",
                            className="info-text"
                        ),
                        html.P(
                            id='filter-description',
                            className="filter-text"
                        )
                    ], className="info-box"),
                    
                    # Heatmap
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='enterprise-heatmap',
                                style={'width': '80%', 'margin': '0 auto'}
                            )
                        ], className="px-0")
                    ], className="g-0"),
                    
                    # Time Series Commentary
                    html.Div([
                        html.P(
                            id='time-series-commentary',
                            className="info-text"
                        )
                    ], className="info-box mt-4"),
                    
                    # Time Series Chart
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='active-customers-chart',
                                style={'width': '80%', 'margin': '0 auto'}
                            )
                        ], className="px-0")
                    ], className="g-0")
                ]),
                dbc.Tab(label="Retention", tab_id="retention", children=[
                    # Filters Row
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Anchor Product"),
                                dcc.Dropdown(
                                    id='retention-anchor-product',
                                    options=[{'label': 'All Products', 'value': 'ALL'}] + [
                                        {'label': product, 'value': product}
                                        for product in DATA['enterprise_cross_sell']['products']
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Customer Type"),
                                dcc.Dropdown(
                                    id='retention-customer-type',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'First Product', 'value': 'NTB'},
                                        {'label': 'Overall', 'value': 'ETB'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Employment Status"),
                                dcc.Dropdown(
                                    id='retention-employment-status',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'Salaried', 'value': 'SALARIED'},
                                        {'label': 'Self Employed', 'value': 'SELF_EMPLOYED'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3),
                            
                            dbc.Col([
                                html.Label("Segment"),
                                dcc.Dropdown(
                                    id='retention-segment',
                                    options=[
                                        {'label': 'All', 'value': 'ALL'},
                                        {'label': 'Mass', 'value': 'MASS'},
                                        {'label': 'Elite', 'value': 'ELITE'},
                                        {'label': 'Select', 'value': 'SELECT'}
                                    ],
                                    value='ALL',
                                    clearable=False
                                )
                            ], width=3)
                        ])
                    ], className="filter-section"),
                    
                    # Description Box
                    html.Div([
                        html.P(
                            "Customer inactivity distribution by duration",
                            className="info-text"
                        ),
                        html.P(
                            id='retention-filter-description',
                            className="filter-text"
                        )
                    ], className="info-box"),
                    
                    # Center the funnel chart
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id='inactive-funnel',
                                    config={
                                        'displayModeBar': False,
                                        'responsive': True
                                    },
                                    style={
                                        'width': '100%',
                                        'height': '100%',
                                        'minHeight': '500px'
                                    }
                                )
                            ], width=10, className="mx-auto")  # Increased width for better visibility
                        ], justify="center", align="center", className="g-0")
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'alignItems': 'center',
                        'width': '100%',
                        'margin': '0 auto'
                    })
                ])
            ],
            id="tabs",
            active_tab="cross-sell"
            )
        ], fluid=True)
    ], className="page-container")
])

@app.callback(
    [Output('enterprise-heatmap', 'figure'),
     Output('active-customers-chart', 'figure'),
     Output('filter-description', 'children'),
     Output('time-series-commentary', 'children')],
    [Input('anchor-product', 'value'),
     Input('customer-type', 'value'),
     Input('employment-status', 'value'),
     Input('segment', 'value')]
)
def update_dashboard(anchor_product, customer_type, employment_status, segment):
    # Prepare filters
    filters = {
        'customer_type': [customer_type],
        'employment_status': [employment_status],
        'segment': [segment]
    }
    
    # Generate data
    heatmap_data = generate_realistic_data(filters)
    time_series_df = generate_time_series_data(filters)
    
    # Create visualizations
    heatmap_fig = create_heatmap(heatmap_data, anchor_product)
    
    time_series_fig = px.line(
        time_series_df,
        x='Date',
        y='Active Customers',
        template='plotly_white'
    )
    time_series_fig.update_traces(line_color='maroon', line_width=2)
    time_series_fig.update_layout(
        height=400,
        margin=dict(t=20, b=40, l=40, r=40),
        yaxis_title="Active Customers",
        xaxis_title="",
        showlegend=False
    )
    
    # Generate text content
    filter_text = get_filter_text(anchor_product, [customer_type], [employment_status], [segment])
    commentary = generate_commentary(time_series_df, filters)
    
    return heatmap_fig, time_series_fig, filter_text, commentary

@app.callback(
    [Output('inactive-funnel', 'figure'),
     Output('retention-filter-description', 'children')],
    [Input('retention-anchor-product', 'value'),
     Input('retention-customer-type', 'value'),
     Input('retention-employment-status', 'value'),
     Input('retention-segment', 'value')]
)
def update_retention_funnel(anchor_product, customer_type, employment_status, segment):
    filters = {
        'anchor_product': anchor_product,
        'customer_type': customer_type,
        'employment_status': employment_status,
        'segment': segment
    }
    
    funnel_data, total_customers = generate_inactive_funnel_data(filters)
    fig = create_inactive_funnel(funnel_data, total_customers)
    
    # Update layout to ensure proper centering and sizing
    fig.update_layout(
        autosize=True,
        height=600,
        margin=dict(t=100, b=20, l=120, r=20)
    )
    
    # Generate filter description
    filter_text = get_retention_filter_text(filters)
    
    return fig, filter_text

if __name__ == '__main__':
    app.run_server(debug=True) 