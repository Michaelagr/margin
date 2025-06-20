import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Margin Overview Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }

    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }

    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }

    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path):
    """Load and process the margin data"""
    try:
        df = pd.read_csv(file_path, sep='\t')

        # Convert contract_date to datetime for better handling
        df['contract_date'] = pd.to_datetime(df['contract_date'], format='%Y%m%d')

        # Calculate total margin DB (sum of both margins * 1.5)
        df['total_margin_db'] = (df['initial_margin'] + df['premium_margin']) * 1.5

        # Round all numeric columns to 2 decimals
        numeric_columns = ['exercise_price', 'initial_margin', 'premium_margin', 'total_margin_db']
        df[numeric_columns] = df[numeric_columns].round(2)

        # Create formatted date string for display
        df['date_display'] = df['contract_date'].dt.strftime('%Y-%m-%d')

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_margin_chart(filtered_data, chart_type="line", log_scale=False):
    """Create interactive margin visualization"""

    if chart_type == "line":
        fig = go.Figure()

        # Add traces for each margin type
        fig.add_trace(go.Scatter(
            x=filtered_data['exercise_price'],
            y=filtered_data['initial_margin'],
            mode='lines+markers',
            name='Initial Margin',
            line=dict(color='dodgerblue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Initial Margin</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_data['exercise_price'],
            y=filtered_data['premium_margin'],
            mode='lines+markers',
            name='Premium Margin',
            line=dict(color='lightskyblue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Premium Margin</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=filtered_data['exercise_price'],
            y=filtered_data['total_margin_db'],
            mode='lines+markers',
            name='Total Margin DB',
            line=dict(color='crimson', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Total Margin DB</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

    else:  # bar chart
        fig = go.Figure()

        width = 0.25
        x_pos = filtered_data['exercise_price']

        fig.add_trace(go.Bar(
            x=x_pos,
            y=filtered_data['initial_margin'],
            name='Initial Margin',
            marker_color='#1f77b4',
            hovertemplate='<b>Initial Margin</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=x_pos,
            y=filtered_data['premium_margin'],
            name='Premium Margin',
            marker_color='#ff7f0e',
            hovertemplate='<b>Premium Margin</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=x_pos,
            y=filtered_data['total_margin_db'],
            name='Total Margin DB',
            marker_color='#d62728',
            hovertemplate='<b>Total Margin DB</b><br>Strike: %{x:,.3f}<br>Value: €%{y:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text="Margin Requirements by Strike Price",
            font=dict(size=20, color='#1f77b4'),
            x=0.5
        ),
        xaxis_title="Exercise Price",
        yaxis_title="Margin Amount (EUR)",
        xaxis=dict(tickformat=',.0f'),
        yaxis=dict(type='log' if log_scale else 'linear'),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )

    return fig


def create_summary_metrics(filtered_data):
    """Create summary metrics for the filtered data"""
    if len(filtered_data) == 0:
        return None

    metrics = {
        'total_positions': len(filtered_data),
        'avg_initial_margin': filtered_data['initial_margin'].mean(),
        'avg_premium_margin': filtered_data['premium_margin'].mean(),
        'avg_total_margin': filtered_data['total_margin_db'].mean(),
        'max_total_margin': filtered_data['total_margin_db'].max(),
        'min_total_margin': filtered_data['total_margin_db'].min(),
        'strike_range': f"{filtered_data['exercise_price'].min()} - {filtered_data['exercise_price'].max()}"
    }

    return metrics


def main():
    # Initialize session state for deals
    if 'deals' not in st.session_state:
        st.session_state.deals = []
    
    # Header
    st.markdown('<div class="main-header">📊 Margin Overview Dashboard</div>', unsafe_allow_html=True)

    # File upload section
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Margin_Result.txt file",
        type=['txt', 'csv'],
        help="Upload your margin data file (tab-separated format)"
    )

    # Load default file if no upload
    if uploaded_file is None:
        st.sidebar.info("💡 Upload your Margin_Result.txt file or place it in the same directory as this script")
        try:
            #df = load_data("C:/Users/mgr/Documents/python/battery-dashboard/Margin games/Margin_Result.txt")
            df = load_data("Margin_Result.txt")
        except:
            st.error("⚠️ Please upload your Margin_Result.txt file using the sidebar")
            return
    else:
        df = load_data(uploaded_file)

    if df is None or len(df) == 0:
        st.error("❌ Could not load data. Please check your file format.")
        return

    # Sidebar filters
    st.sidebar.header("🔧 Filters")

    # Date selection
    available_dates = sorted(df['date_display'].unique())
    selected_date = st.sidebar.selectbox(
        "📅 Select Contract Date:",
        available_dates,
        help="Choose the option expiry date"
    )

    # Call/Put selection with Put as default
    available_options = sorted(df['call_put_flag'].unique())
    option_labels = {'C': 'Call Options', 'P': 'Put Options'}
    default_index = available_options.index('P') if 'P' in available_options else 0
    selected_option = st.sidebar.selectbox(
        "📈 Select Option Type:",
        available_options,
        index=default_index,
        format_func=lambda x: option_labels.get(x, x),
        help="Choose between Call or Put options"
    )

    # Chart type selection
    chart_type = st.sidebar.radio(
        "📊 Chart Type:",
        ["line", "bar"],
        format_func=lambda x: "Line Chart" if x == "line" else "Bar Chart"
    )
    
    # Logarithmic scale toggle
    log_scale = st.sidebar.checkbox("📈 Logarithmic Y-axis", value=False)

    # Filter data
    filtered_df = df[
        (df['date_display'] == selected_date) &
        (df['call_put_flag'] == selected_option)
        ].copy()

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Sort by exercise price for better visualization
    filtered_df = filtered_df.sort_values('exercise_price')

    # Strike price range filter (main area)
    st.subheader(f"📊 {option_labels.get(selected_option, selected_option)} - {selected_date}")

    col_range1, col_range2 = st.columns(2)
    with col_range1:
        min_strike = st.number_input(
            "From Strike Price:",
            min_value=float(filtered_df['exercise_price'].min()),
            max_value=float(filtered_df['exercise_price'].max()),
            value=float(filtered_df['exercise_price'].min()),
            format="%.3f"
        )
    with col_range2:
        max_strike = st.number_input(
            "To Strike Price:",
            min_value=float(filtered_df['exercise_price'].min()),
            max_value=float(filtered_df['exercise_price'].max()),
            value=float(filtered_df['exercise_price'].max()),
            format="%.3f"
        )

    # Filter by strike price range
    filtered_df = filtered_df[
        (filtered_df['exercise_price'] >= min_strike) &
        (filtered_df['exercise_price'] <= max_strike)
        ]


    # Create and display chart
    fig = create_margin_chart(filtered_df, chart_type, log_scale)
    st.plotly_chart(fig, use_container_width=True)

    # My Deals Section
    st.subheader("💰 My Deals")
    
    # Add new deal
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        deal_date = st.selectbox("Date:", available_dates, key="deal_date")
    with col2:
        deal_type = st.selectbox("Type:", available_options, format_func=lambda x: option_labels.get(x, x), key="deal_type")
    with col3:
        deal_strike = st.number_input("Strike:", min_value=0.0, step=100.0, format="%.3f", key="deal_strike")
    with col4:
        deal_quantity = st.number_input("Quantity:", min_value=1, step=1, key="deal_quantity")
    
    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add Deal"):
            # Find margin data for this deal
            deal_margin_data = df[
                (df['date_display'] == deal_date) & 
                (df['call_put_flag'] == deal_type) & 
                (df['exercise_price'] == deal_strike)
            ]
            
            if len(deal_margin_data) > 0:
                margin_per_contract = deal_margin_data.iloc[0]['total_margin_db']
                total_margin = margin_per_contract * deal_quantity
                
                new_deal = {
                    'date': deal_date,
                    'type': deal_type,
                    'strike': deal_strike,
                    'quantity': deal_quantity,
                    'margin_per_contract': margin_per_contract,
                    'total_margin': total_margin
                }
                st.session_state.deals.append(new_deal)
                st.success("Deal added!")
            else:
                st.error("No margin data found for this combination")
    
    with col_clear:
        if st.button("🗑️ Clear All Deals"):
            st.session_state.deals = []
            st.success("All deals cleared!")
    
    # Display deals and total margin
    if st.session_state.deals:
        deals_df = pd.DataFrame(st.session_state.deals)
        deals_df['Type'] = deals_df['type'].map(option_labels)
        
        display_deals = deals_df[['date', 'Type', 'strike', 'quantity', 'margin_per_contract', 'total_margin']].copy()
        display_deals.columns = ['Date', 'Type', 'Strike', 'Quantity', 'Margin/Contract (€)', 'Total Margin (€)']
        
        st.dataframe(display_deals, use_container_width=True, hide_index=True)
        
        total_portfolio_margin = deals_df['total_margin'].sum()
        st.metric("🎯 Total Portfolio Margin Requirement", f"€{total_portfolio_margin:,.2f}")
    else:
        st.info("No deals added yet. Add your first deal above!")

    st.markdown("---")

    # Detailed data table
    st.subheader("📋 Detailed Data")

    # Display options
    col1, col2 = st.columns(2)
    with col1:
        show_all_columns = st.checkbox("Show all columns", value=False)
    with col2:
        max_rows = st.selectbox("Rows to display:", [10, 25, 50, 100], index=1)

    # Prepare display dataframe
    display_df = filtered_df.copy()

    if not show_all_columns:
        display_df = display_df[['exercise_price', 'initial_margin', 'premium_margin', 'total_margin_db']]
        display_df.columns = ['Strike Price', 'Initial Margin (€)', 'Premium Margin (€)', 'Total Margin DB (€)']

    # Display table with formatting
    st.dataframe(
        display_df.head(max_rows),
        use_container_width=True,
        hide_index=True
    )

    # Download option
    if st.button("📥 Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"margin_data_{selected_date}_{selected_option}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "💡 Total Margin DB = (Initial Margin + Premium Margin) × 1.5 | Built with Streamlit & Plotly"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
