import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Revenue Forecasting App", layout="wide")

def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        # Direct Sales
        'direct_sales_cycle': 6,
        'direct_ticket_small': 75000,
        'direct_ticket_medium': 150000,
        'direct_ticket_large': 300000,
        'direct_pct_small': 50,
        'direct_pct_medium': 30,
        'direct_pct_large': 20,
        'direct_activation': 2,
        'direct_onboarding_small': 15000,
        'direct_onboarding_medium': 30000,
        'direct_onboarding_large': 60000,
        'direct_growth_rate': 30,
        'direct_churn_rate': 10,
        
        # Premium Partners
        'premium_partners_count': 3,
        'premium_contract_value': 500000,
        'premium_subsidy_year1': 60,
        'premium_subsidy_after': 20,
        'premium_leads_per_month': 8,
        
        # Standard Partners
        'standard_partners_count': 5,
        'standard_contract_value': 200000,
        'standard_subsidy_year1': 50,
        'standard_subsidy_after': 30,
        'standard_leads_per_month': 3,
        
        # Partner General
        'partner_activation': 2,
        'partner_conversion_rate': 15,
        'partner_growth_rate': 30,
        'partner_churn_rate': 5,
        
        # Data Owners
        'data_requests_small': 10000,
        'data_requests_medium': 50000,
        'data_requests_large': 100000,
        'data_price_small': 0.50,
        'data_price_medium': 1.50,
        'data_price_large': 2.50,
        'data_activation': 2,
        'data_growth_rate': 20,
        'data_churn_rate': 10,
        
        # Lead Generation
        'lead_cost': 200,
        'initial_budget': 2000,
        'budget_growth_rate': 5,
        'lead_conversion_rate': 15,
        'starting_leads': 100
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class RevenueForecaster:
    def __init__(self, params):
        self.params = params
        self.start_date = datetime(2026, 1, 1)
        self.end_date = datetime(2028, 12, 31)
        self.months = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        
    def calculate_weighted_average(self, values, percentages):
        """Calculate weighted average for ticket sizes"""
        return sum(v * p / 100 for v, p in zip(values, percentages))
    
    def apply_growth(self, base_value, growth_rate_annual, month_index):
        """Apply annual growth rate monthly"""
        monthly_growth = (1 + growth_rate_annual / 100) ** (1/12) - 1
        return base_value * ((1 + monthly_growth) ** month_index)
    
    def calculate_monthly_churn(self, customers, annual_churn_rate):
        """Calculate monthly customer churn"""
        monthly_churn_rate = 1 - (1 - annual_churn_rate / 100) ** (1/12)
        return customers * monthly_churn_rate
    
    def generate_forecast(self):
        """Generate complete revenue forecast"""
        results = []
        
        # Initialize customer tracking
        direct_customers = 0
        premium_customers = {}  # Track by partner and deal date
        standard_customers = {}  # Track by partner and deal date
        data_customers = 0
        
        # Lead pipeline tracking
        direct_leads_pipeline = []
        premium_leads_pipeline = []
        standard_leads_pipeline = []
        data_leads_pipeline = []
        
        for month_idx, month in enumerate(self.months):
            month_data = {'Month': month.strftime('%Y-%m')}
            
            # Calculate current budget with growth
            current_budget = self.apply_growth(
                self.params['initial_budget'], 
                self.params['budget_growth_rate'] * 12,  # Convert monthly to annual for consistency
                month_idx
            )
            
            # Generate leads based on budget
            leads_generated = current_budget / self.params['lead_cost']
            if month_idx == 0:
                leads_generated += self.params['starting_leads']
            
            # Apply growth to lead generation capacity
            leads_generated = self.apply_growth(leads_generated, self.params['direct_growth_rate'], month_idx)
            
            # Convert leads to customers with activation delay
            converted_customers = leads_generated * (self.params['lead_conversion_rate'] / 100)
            
            # Add to pipeline with activation delay
            activation_month = month_idx + self.params['direct_activation']
            if activation_month < len(self.months):
                direct_leads_pipeline.append({
                    'activation_month': activation_month,
                    'customers': converted_customers * 0.6  # 60% go to direct sales
                })
                
                # Premium partners
                for partner in range(self.params['premium_partners_count']):
                    partner_leads = self.apply_growth(
                        self.params['premium_leads_per_month'], 
                        self.params['partner_growth_rate'], 
                        month_idx
                    )
                    partner_customers = partner_leads * (self.params['partner_conversion_rate'] / 100)
                    
                    premium_leads_pipeline.append({
                        'activation_month': activation_month,
                        'customers': partner_customers,
                        'partner_id': partner,
                        'deal_date': month
                    })
                
                # Standard partners  
                for partner in range(self.params['standard_partners_count']):
                    partner_leads = self.apply_growth(
                        self.params['standard_leads_per_month'], 
                        self.params['partner_growth_rate'], 
                        month_idx
                    )
                    partner_customers = partner_leads * (self.params['partner_conversion_rate'] / 100)
                    
                    standard_leads_pipeline.append({
                        'activation_month': activation_month,
                        'customers': partner_customers,
                        'partner_id': partner,
                        'deal_date': month
                    })
                
                # Data owners
                data_leads_pipeline.append({
                    'activation_month': activation_month,
                    'customers': converted_customers * 0.1  # 10% go to data owners
                })
            
            # Activate customers from pipeline
            direct_new_customers = sum([p['customers'] for p in direct_leads_pipeline if p['activation_month'] == month_idx])
            direct_customers += direct_new_customers
            
            # Activate premium partner customers
            for pipeline_item in [p for p in premium_leads_pipeline if p['activation_month'] == month_idx]:
                partner_key = f"premium_{pipeline_item['partner_id']}"
                if partner_key not in premium_customers:
                    premium_customers[partner_key] = []
                premium_customers[partner_key].append({
                    'customers': pipeline_item['customers'],
                    'deal_date': pipeline_item['deal_date'],
                    'start_month': month
                })
            
            # Activate standard partner customers
            for pipeline_item in [p for p in standard_leads_pipeline if p['activation_month'] == month_idx]:
                partner_key = f"standard_{pipeline_item['partner_id']}"
                if partner_key not in standard_customers:
                    standard_customers[partner_key] = []
                standard_customers[partner_key].append({
                    'customers': pipeline_item['customers'],
                    'deal_date': pipeline_item['deal_date'],
                    'start_month': month
                })
            
            # Activate data customers
            data_new_customers = sum([p['customers'] for p in data_leads_pipeline if p['activation_month'] == month_idx])
            data_customers += data_new_customers
            
            # Apply churn
            direct_churn = self.calculate_monthly_churn(direct_customers, self.params['direct_churn_rate'])
            direct_customers = max(0, direct_customers - direct_churn)
            
            data_churn = self.calculate_monthly_churn(data_customers, self.params['data_churn_rate'])
            data_customers = max(0, data_customers - data_churn)
            
            # Calculate revenues
            # Direct Sales Revenue
            direct_ticket_avg = self.calculate_weighted_average(
                [self.params['direct_ticket_small'], self.params['direct_ticket_medium'], self.params['direct_ticket_large']],
                [self.params['direct_pct_small'], self.params['direct_pct_medium'], self.params['direct_pct_large']]
            )
            direct_revenue = direct_customers * (direct_ticket_avg / 12)
            
            # Premium Partner Revenue
            premium_revenue = 0
            for partner_key, deals in premium_customers.items():
                for deal in deals:
                    months_since_deal = (month.year - deal['deal_date'].year) * 12 + (month.month - deal['deal_date'].month)
                    if months_since_deal < 12:
                        # First year - apply subsidy
                        revenue_share = (100 - self.params['premium_subsidy_year1']) / 100
                    else:
                        # After first year - apply different subsidy
                        revenue_share = (100 - self.params['premium_subsidy_after']) / 100
                    
                    premium_revenue += deal['customers'] * (self.params['premium_contract_value'] / 12) * revenue_share
            
            # Standard Partner Revenue
            standard_revenue = 0
            for partner_key, deals in standard_customers.items():
                for deal in deals:
                    months_since_deal = (month.year - deal['deal_date'].year) * 12 + (month.month - deal['deal_date'].month)
                    if months_since_deal < 12:
                        # First year - apply subsidy
                        revenue_share = (100 - self.params['standard_subsidy_year1']) / 100
                    else:
                        # After first year - apply different subsidy
                        revenue_share = (100 - self.params['standard_subsidy_after']) / 100
                    
                    standard_revenue += deal['customers'] * (self.params['standard_contract_value'] / 12) * revenue_share
            
            # Data Owners Revenue
            data_avg_price = self.calculate_weighted_average(
                [self.params['data_price_small'], self.params['data_price_medium'], self.params['data_price_large']],
                [33.33, 33.33, 33.33]  # Equal distribution assumption
            )
            data_avg_requests = self.calculate_weighted_average(
                [self.params['data_requests_small'], self.params['data_requests_medium'], self.params['data_requests_large']],
                [33.33, 33.33, 33.33]  # Equal distribution assumption
            )
            data_revenue = data_customers * data_avg_requests * data_avg_price
            
            # Store results
            month_data.update({
                'Direct_Sales_Customers': round(direct_customers, 1),
                'Direct_Sales_Revenue': round(direct_revenue, 2),
                'Premium_Partner_Revenue': round(premium_revenue, 2),
                'Standard_Partner_Revenue': round(standard_revenue, 2),
                'Data_Owners_Customers': round(data_customers, 1),
                'Data_Owners_Revenue': round(data_revenue, 2),
                'Total_Revenue': round(direct_revenue + premium_revenue + standard_revenue + data_revenue, 2)
            })
            
            results.append(month_data)
        
        return pd.DataFrame(results)

def main():
    st.title("🚀 Revenue Forecasting App")
    st.markdown("*Forecast revenues across Direct Sales, Partner Channels, and Data Owners (Jan 2026 - Dec 2028)*")
    
    initialize_session_state()
    
    # Sidebar controls
    st.sidebar.title("📊 Simulation Controls")
    
    # Direct Sales Parameters
    st.sidebar.subheader("🎯 Direct Sales")
    st.session_state.direct_sales_cycle = st.sidebar.slider("Sales Cycle (months)", 1, 12, st.session_state.direct_sales_cycle)
    st.session_state.direct_ticket_small = st.sidebar.number_input("Small Ticket Size ($)", 10000, 200000, st.session_state.direct_ticket_small, step=5000)
    st.session_state.direct_ticket_medium = st.sidebar.number_input("Medium Ticket Size ($)", 50000, 500000, st.session_state.direct_ticket_medium, step=10000)
    st.session_state.direct_ticket_large = st.sidebar.number_input("Large Ticket Size ($)", 100000, 1000000, st.session_state.direct_ticket_large, step=25000)
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.session_state.direct_pct_small = st.number_input("Small %", 0, 100, st.session_state.direct_pct_small)
    with col2:
        st.session_state.direct_pct_medium = st.number_input("Med %", 0, 100, st.session_state.direct_pct_medium)
    with col3:
        st.session_state.direct_pct_large = st.number_input("Large %", 0, 100, st.session_state.direct_pct_large)
    
    st.session_state.direct_growth_rate = st.sidebar.slider("Direct Growth Rate (% annual)", 0, 100, st.session_state.direct_growth_rate)
    st.session_state.direct_churn_rate = st.sidebar.slider("Direct Churn Rate (% annual)", 0, 30, st.session_state.direct_churn_rate)
    
    # Premium Partners
    st.sidebar.subheader("👑 Embedded Partners")
    st.session_state.premium_partners_count = st.sidebar.slider("Number of Premium Partners", 1, 10, st.session_state.premium_partners_count)
    st.session_state.premium_contract_value = st.sidebar.number_input("Premium Contract Value ($)", 100000, 2000000, st.session_state.premium_contract_value, step=50000)
    st.session_state.premium_subsidy_year1 = st.sidebar.slider("Premium Subsidy Year 1 (%)", 0, 90, st.session_state.premium_subsidy_year1)
    st.session_state.premium_subsidy_after = st.sidebar.slider("Premium Subsidy After Year 1 (%)", 0, 90, st.session_state.premium_subsidy_after)
    st.session_state.premium_leads_per_month = st.sidebar.slider("Premium Leads/Partner/Month", 1, 20, st.session_state.premium_leads_per_month)
    
    # Standard Partners
    st.sidebar.subheader("🤝 Reseller Partners")
    st.session_state.standard_partners_count = st.sidebar.slider("Number of Standard Partners", 1, 20, st.session_state.standard_partners_count)
    st.session_state.standard_contract_value = st.sidebar.number_input("Standard Contract Value ($)", 50000, 1000000, st.session_state.standard_contract_value, step=25000)
    st.session_state.standard_subsidy_year1 = st.sidebar.slider("Standard Subsidy Year 1 (%)", 0, 90, st.session_state.standard_subsidy_year1)
    st.session_state.standard_subsidy_after = st.sidebar.slider("Standard Subsidy After Year 1 (%)", 0, 90, st.session_state.standard_subsidy_after)
    st.session_state.standard_leads_per_month = st.sidebar.slider("Standard Leads/Partner/Month", 1, 15, st.session_state.standard_leads_per_month)
    
    # Partner General
    st.session_state.partner_conversion_rate = st.sidebar.slider("Partner Conversion Rate (%)", 5, 50, st.session_state.partner_conversion_rate)
    st.session_state.partner_growth_rate = st.sidebar.slider("Partner Growth Rate (% annual)", 0, 100, st.session_state.partner_growth_rate)
    
    # Data Owners
    st.sidebar.subheader("📊 Data Owners")
    st.session_state.data_requests_small = st.sidebar.number_input("Small Volume Requests/Month", 1000, 50000, st.session_state.data_requests_small, step=1000)
    st.session_state.data_requests_medium = st.sidebar.number_input("Medium Volume Requests/Month", 10000, 200000, st.session_state.data_requests_medium, step=5000)
    st.session_state.data_requests_large = st.sidebar.number_input("Large Volume Requests/Month", 50000, 500000, st.session_state.data_requests_large, step=10000)
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.session_state.data_price_small = st.number_input("Small Price ($)", 0.1, 5.0, st.session_state.data_price_small, step=0.1)
    with col2:
        st.session_state.data_price_medium = st.number_input("Med Price ($)", 0.5, 10.0, st.session_state.data_price_medium, step=0.1)
    with col3:
        st.session_state.data_price_large = st.number_input("Large Price ($)", 1.0, 20.0, st.session_state.data_price_large, step=0.1)
    
    st.session_state.data_growth_rate = st.sidebar.slider("Data Growth Rate (% annual)", 0, 100, st.session_state.data_growth_rate)
    st.session_state.data_churn_rate = st.sidebar.slider("Data Churn Rate (% annual)", 0, 30, st.session_state.data_churn_rate)
    
    # Lead Generation
    st.sidebar.subheader("💰 Lead Generation")
    st.session_state.lead_cost = st.sidebar.number_input("Lead Cost ($)", 50, 1000, st.session_state.lead_cost, step=25)
    st.session_state.initial_budget = st.sidebar.number_input("Initial Monthly Budget ($)", 500, 10000, st.session_state.initial_budget, step=100)
    st.session_state.budget_growth_rate = st.sidebar.slider("Budget Growth Rate (% monthly)", 0, 20, st.session_state.budget_growth_rate)
    st.session_state.lead_conversion_rate = st.sidebar.slider("Lead Conversion Rate (%)", 5, 50, st.session_state.lead_conversion_rate)
    st.session_state.starting_leads = st.sidebar.number_input("Starting Leads (Jan 2026)", 0, 1000, st.session_state.starting_leads, step=10)
    
    # Generate forecast
    params = {k: v for k, v in st.session_state.items() if not k.startswith('_')}
    forecaster = RevenueForecaster(params)
    
    if st.button("🚀 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            df = forecaster.generate_forecast()
            
            # Display results
            st.success("🎉 Forecast generated successfully!")
            
            # Create horizontal tables
            df['Year'] = pd.to_datetime(df['Month']).dt.year
            df['Month_Name'] = pd.to_datetime(df['Month']).dt.strftime('%b %Y')
            
            # Annual summaries - Horizontal Layout
            annual_summary = df.groupby('Year').agg({
                'Direct_Sales_Revenue': 'sum',
                'Premium_Partner_Revenue': 'sum',
                'Standard_Partner_Revenue': 'sum',
                'Data_Owners_Revenue': 'sum',
                'Total_Revenue': 'sum'
            }).round(0)
            
            # Calculate percentage distribution for each year
            annual_percentages = annual_summary.div(annual_summary['Total_Revenue'], axis=0) * 100
            annual_percentages = annual_percentages.drop('Total_Revenue', axis=1).round(1)  # Remove total from percentages
            
            # Transpose for horizontal view
            annual_horizontal = annual_summary.T
            annual_horizontal.columns = [f'{int(year)}' for year in annual_horizontal.columns]
            
            # Create percentage table for display
            annual_pct_horizontal = annual_percentages.T
            annual_pct_horizontal.columns = [f'{int(year)}' for year in annual_pct_horizontal.columns]
            
            st.subheader("📈 Annual Revenue Summary")
            
            # Display revenue amounts
            st.write("**Revenue by Channel ($)**")
            annual_styled = annual_horizontal.style.format("${:,.0f}").set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6'), 
                                             ('color', '#262730'), 
                                             ('font-weight', 'bold'),
                                             ('text-align', 'center'),
                                             ('padding', '12px')]},
                {'selector': 'td', 'props': [('text-align', 'center'), 
                                             ('padding', '10px'),
                                             ('font-weight', 'bold')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
                {'selector': 'tr:last-child', 'props': [('background-color', '#e8f4f8'), 
                                                        ('font-weight', 'bold'),
                                                        ('border-top', '2px solid #1f77b4')]},
                {'selector': '', 'props': [('border-collapse', 'collapse'), 
                                          ('margin', '25px 0'),
                                          ('font-size', '16px'),
                                          ('border-radius', '5px'),
                                          ('overflow', 'hidden'),
                                          ('box-shadow', '0 0 20px rgba(0,0,0,0.15)')]}
            ])
            
            st.dataframe(annual_styled, use_container_width=True, height=220)
            
            # Display percentage distribution
            st.write("**Channel Distribution (%)**")
            annual_pct_styled = annual_pct_horizontal.style.format("{:.1f}%").set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f8f0'), 
                                             ('color', '#262730'), 
                                             ('font-weight', 'bold'),
                                             ('text-align', 'center'),
                                             ('padding', '12px')]},
                {'selector': 'td', 'props': [('text-align', 'center'), 
                                             ('padding', '10px'),
                                             ('font-weight', 'bold')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9fff9')]},
                {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
                {'selector': '', 'props': [('border-collapse', 'collapse'), 
                                          ('margin', '25px 0'),
                                          ('font-size', '16px'),
                                          ('border-radius', '5px'),
                                          ('overflow', 'hidden'),
                                          ('box-shadow', '0 0 15px rgba(0,0,0,0.1)')]}
            ])
            
            st.dataframe(annual_pct_styled, use_container_width=True, height=180)
            
            # Add total row with highlighting
            total_by_year = annual_summary['Total_Revenue']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Total 2026", f"${total_by_year[2026]:,.0f}")
            with col2:
                st.metric("💰 Total 2027", f"${total_by_year[2027]:,.0f}", 
                         delta=f"+${(total_by_year[2027] - total_by_year[2026]):,.0f}")
            with col3:
                st.metric("💰 Total 2028", f"${total_by_year[2028]:,.0f}", 
                         delta=f"+${(total_by_year[2028] - total_by_year[2027]):,.0f}")
            with col4:
                three_year_total = total_by_year.sum()
                st.metric("🚀 3-Year Total", f"${three_year_total:,.0f}")
            
            st.divider()
            
            # Add Charts Section
            st.subheader("📊 Revenue Visualization")
            
            # Chart tabs
            chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["📈 Monthly Trends", "🥧 Channel Mix", "📊 Annual Comparison", "📈 Growth Trends"])
            
            with chart_tab1:
                # Monthly revenue trends
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig_monthly = go.Figure()
                
                # Add traces for each channel
                fig_monthly.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Direct_Sales_Revenue'],
                    mode='lines+markers', name='Direct Sales',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6)
                ))
                
                fig_monthly.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Premium_Partner_Revenue'],
                    mode='lines+markers', name='Premium Partners',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6)
                ))
                
                fig_monthly.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Standard_Partner_Revenue'],
                    mode='lines+markers', name='Standard Partners',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=6)
                ))
                
                fig_monthly.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Data_Owners_Revenue'],
                    mode='lines+markers', name='Data Owners',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=6)
                ))
                
                # Add total revenue line (thicker)
                fig_monthly.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Total_Revenue'],
                    mode='lines+markers', name='Total Revenue',
                    line=dict(color='#17becf', width=4),
                    marker=dict(size=8)
                ))
                
                fig_monthly.update_layout(
                    title='Monthly Revenue Trends by Channel',
                    xaxis_title='Month',
                    yaxis_title='Revenue ($)',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    yaxis=dict(tickformat='$,.0f'),
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with chart_tab2:
                # Channel mix pie charts for each year
                col1, col2, col3 = st.columns(3)
                
                for idx, year in enumerate([2026, 2027, 2028]):
                    # Get year data from the original annual_summary (before transpose)
                    year_revenue = annual_summary[annual_summary.index == year].iloc[0]
                    year_data = year_revenue.drop('Total_Revenue')
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Direct Sales', 'Premium Partners', 'Standard Partners', 'Data Owners'],
                        values=year_data.values,
                        hole=0.3,
                        textinfo='label+percent',
                        textposition='outside',
                        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    )])
                    
                    fig_pie.update_layout(
                        title=f'{year} Channel Mix',
                        height=400,
                        showlegend=False
                    )
                    
                    if idx == 0:
                        col1.plotly_chart(fig_pie, use_container_width=True)
                    elif idx == 1:
                        col2.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        col3.plotly_chart(fig_pie, use_container_width=True)
            
            with chart_tab3:
                # Annual comparison bar chart
                fig_annual = go.Figure()
                
                channels = ['Direct_Sales_Revenue', 'Premium_Partner_Revenue', 'Standard_Partner_Revenue', 'Data_Owners_Revenue']
                channel_labels = ['Direct Sales', 'Premium Partners', 'Standard Partners', 'Data Owners']
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                for i, (channel, label, color) in enumerate(zip(channels, channel_labels, colors)):
                    # Get values for each year for this channel
                    values = [annual_summary.loc[annual_summary.index == year, channel].iloc[0] for year in [2026, 2027, 2028]]
                    
                    fig_annual.add_trace(go.Bar(
                        name=label,
                        x=['2026', '2027', '2028'],
                        y=values,
                        marker_color=color,
                        text=[f'${val:,.0f}' for val in values],
                        textposition='outside'
                    ))
                
                fig_annual.update_layout(
                    title='Annual Revenue Comparison by Channel',
                    xaxis_title='Year',
                    yaxis_title='Revenue ($)',
                    barmode='group',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(tickformat='$,.0f')
                )
                
                st.plotly_chart(fig_annual, use_container_width=True)
            
            with chart_tab4:
                # Stacked area chart showing cumulative growth
                fig_area = go.Figure()
                
                fig_area.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Direct_Sales_Revenue'],
                    mode='lines', name='Direct Sales',
                    stackgroup='one', fill='tonexty',
                    line=dict(color='#1f77b4')
                ))
                
                fig_area.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Premium_Partner_Revenue'],
                    mode='lines', name='Premium Partners',
                    stackgroup='one', fill='tonexty',
                    line=dict(color='#ff7f0e')
                ))
                
                fig_area.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Standard_Partner_Revenue'],
                    mode='lines', name='Standard Partners',
                    stackgroup='one', fill='tonexty',
                    line=dict(color='#2ca02c')
                ))
                
                fig_area.add_trace(go.Scatter(
                    x=df['Month_Name'], y=df['Data_Owners_Revenue'],
                    mode='lines', name='Data Owners',
                    stackgroup='one', fill='tonexty',
                    line=dict(color='#d62728')
                ))
                
                fig_area.update_layout(
                    title='Revenue Growth - Stacked Area Chart',
                    xaxis_title='Month',
                    yaxis_title='Cumulative Revenue ($)',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    yaxis=dict(tickformat='$,.0f'),
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig_area, use_container_width=True)
            
            st.divider()
            
            # Monthly detail - Horizontal Layout  
            st.subheader("📅 Monthly Forecast Details")
            
            # Create pivot table for horizontal monthly view
            monthly_data = df[['Month_Name', 'Direct_Sales_Revenue', 'Premium_Partner_Revenue', 
                              'Standard_Partner_Revenue', 'Data_Owners_Revenue', 'Total_Revenue']].copy()
            
            # Transpose monthly data
            monthly_horizontal = monthly_data.set_index('Month_Name').T
            
            # Custom styling for monthly table
            monthly_styled = monthly_horizontal.style.format("${:,.0f}").set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#1f77b4'), 
                                             ('color', 'white'), 
                                             ('font-weight', 'bold'),
                                             ('text-align', 'center'),
                                             ('padding', '8px'),
                                             ('font-size', '12px')]},
                {'selector': 'td', 'props': [('text-align', 'center'), 
                                             ('padding', '6px'),
                                             ('font-size', '11px')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f8f9fa')]},
                {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
                {'selector': 'tr:last-child', 'props': [('background-color', '#e8f4f8'), 
                                                        ('font-weight', 'bold'),
                                                        ('border-top', '2px solid #1f77b4')]},
                {'selector': '', 'props': [('border-collapse', 'collapse'), 
                                          ('margin', '15px 0'),
                                          ('border-radius', '8px'),
                                          ('overflow', 'hidden'),
                                          ('box-shadow', '0 4px 6px rgba(0,0,0,0.1)')]}
            ]).set_properties(**{
                'border': '1px solid #ddd'
            })
            
            st.dataframe(monthly_styled, use_container_width=True, height=200)
            
            # Add some key insights
            st.subheader("🔍 Key Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                peak_month = df.loc[df['Total_Revenue'].idxmax()]
                st.info(f"📊 **Peak Revenue Month:** {peak_month['Month_Name']} (${peak_month['Total_Revenue']:,.0f})")
                
                # Calculate YoY growth 2026 to 2027
                revenue_2026 = total_by_year[2026]
                revenue_2027 = total_by_year[2027]
                yoy_2026_2027 = ((revenue_2027 - revenue_2026) / revenue_2026) * 100
                st.info(f"📈 **YoY Growth 2026→2027:** {yoy_2026_2027:,.1f}%")
                
            with insights_col2:
                avg_monthly = df['Total_Revenue'].mean()
                st.info(f"💰 **Average Monthly Revenue:** ${avg_monthly:,.0f}")
                
                # Calculate YoY growth 2027 to 2028
                revenue_2028 = total_by_year[2028]
                yoy_2027_2028 = ((revenue_2028 - revenue_2027) / revenue_2027) * 100
                st.info(f"🚀 **YoY Growth 2027→2028:** {yoy_2027_2028:,.1f}%")
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download Forecast as CSV",
                data=csv,
                file_name="revenue_forecast.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
