import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- App Configuration ---
st.set_page_config(
    page_title="Share Buyback Pre-Trade Simulator",
    page_icon="🏦",
    layout="wide"
)

# --- Simulation Engine ---
def run_gbm_simulation(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int,
    n_paths: int,
    trading_days_per_year: int = 252,
) -> np.ndarray:
    """
    Generates multiple asset price paths using Geometric Brownian Motion.
    Returns a 2D array of price paths with shape (n_paths, n_days + 1).
    """
    dt = 1 / trading_days_per_year
    # The Ito-corrected drift for the log-process
    m = mu - 0.5 * sigma**2
    
    # Generate random shocks for all paths and steps at once
    random_shocks = np.random.normal(0.0, 1.0, size=(n_paths, n_days))
    
    # Calculate daily log returns for all paths
    log_returns = (m * dt) + (sigma * np.sqrt(dt) * random_shocks)
    
    # Calculate cumulative log returns and exponentiate to get prices
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    price_paths = s0 * np.exp(cumulative_log_returns)
    
    # Prepend the initial price S0 to each path
    initial_prices = np.full((n_paths, 1), s0)
    full_price_paths = np.hstack([initial_prices, price_paths])
    
    return full_price_paths

# --- Strategy Analysis ---
def analyze_dca_strategy(price_paths: np.ndarray, daily_investment: float) -> pd.DataFrame:
    """Analyzes the daily dollar-cost averaging strategy."""
    n_paths, n_steps = price_paths.shape
    n_days = n_steps - 1
    investment_prices = price_paths[:, 1:]
    shares_acquired_daily = daily_investment / investment_prices
    total_shares_acquired = np.sum(shares_acquired_daily, axis=1)
    total_usd_invested = daily_investment * n_days
    avg_execution_price = total_usd_invested / total_shares_acquired
    simple_avg_price = np.mean(investment_prices, axis=1)
    total_performance_bps = (1 - (avg_execution_price / simple_avg_price)) * 10000
    terminal_price = price_paths[:, -1]
    
    return pd.DataFrame({
        'path_id': range(1, n_paths + 1),
        'terminal_price': terminal_price,
        'total_usd_invested': total_usd_invested,
        'total_shares_acquired': total_shares_acquired,
        'avg_execution_price': avg_execution_price,
        'simple_avg_price': simple_avg_price,
        'total_performance_bps': total_performance_bps,
        'timing_alpha_bps': 0, # No timing alpha for DCA
        'duration_alpha_bps': 0, # No duration alpha for DCA
        'actual_investment_days': n_days
    })

def analyze_adaptive_strategy(price_paths: np.ndarray, total_investment: float, base_n_days: int, min_investment_days: int, max_investment_days: int, modulation_factor: float) -> pd.DataFrame:
    """Analyzes the adaptive (mean-reversion) investment strategy using a full running average."""
    n_paths, n_steps = price_paths.shape
    results = []
    base_daily_investment = total_investment / base_n_days

    for i in range(n_paths):
        path = price_paths[i, :]
        
        # --- Actual (Variable Duration) Execution ---
        remaining_investment = total_investment
        total_shares = 0
        days_invested = 0
        total_cost = 0
        for t in range(1, len(path)):
            if days_invested >= max_investment_days or (remaining_investment <= 1e-6 and days_invested >= min_investment_days): break
            hist_prices = path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else path[t]
            modulation = modulation_factor if path[t] < running_avg else 1 / modulation_factor
            actual_investment = min(base_daily_investment * modulation, remaining_investment)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            total_shares += shares_bought
            total_cost += actual_investment
            remaining_investment -= actual_investment
            days_invested += 1
        
        avg_execution_price = total_cost / total_shares if total_shares > 0 else 0
        simple_avg_price = np.mean(path[1:days_invested+1]) if days_invested > 0 else 0
        total_performance_bps = (1 - (avg_execution_price / simple_avg_price)) * 10000 if simple_avg_price > 0 else 0

        # --- Fixed Duration Benchmark for Timing Alpha ---
        fixed_duration = base_n_days
        fixed_total_cost = 0
        fixed_total_shares = 0
        remaining_investment_fixed = total_investment
        for t in range(1, fixed_duration + 1):
            if t >= len(path): break
            hist_prices = path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else path[t]
            modulation = modulation_factor if path[t] < running_avg else 1 / modulation_factor
            daily_target = (remaining_investment_fixed) / (fixed_duration - (t - 1)) if (fixed_duration - (t-1)) > 0 else remaining_investment_fixed
            actual_investment = min(daily_target * modulation, remaining_investment_fixed)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            fixed_total_shares += shares_bought
            fixed_total_cost += actual_investment
            remaining_investment_fixed -= actual_investment


        fixed_avg_exec_price = fixed_total_cost / fixed_total_shares if fixed_total_shares > 0 else 0
        fixed_simple_avg_price = np.mean(path[1:fixed_duration+1]) if fixed_duration > 0 else 0
        timing_alpha_bps = (1 - (fixed_avg_exec_price / fixed_simple_avg_price)) * 10000 if fixed_simple_avg_price > 0 else 0
        duration_alpha_bps = total_performance_bps - timing_alpha_bps
        
        results.append({
            'path_id': i + 1, 'terminal_price': path[-1], 'total_usd_invested': total_cost,
            'total_shares_acquired': total_shares, 'avg_execution_price': avg_execution_price,
            'simple_avg_price': simple_avg_price, 'total_performance_bps': total_performance_bps,
            'timing_alpha_bps': timing_alpha_bps, 'duration_alpha_bps': duration_alpha_bps,
            'actual_investment_days': days_invested
        })

    return pd.DataFrame(results)

def analyze_dynamic_adaptive_strategy(price_paths: np.ndarray, total_investment: float, min_investment_days: int, max_investment_days: int) -> pd.DataFrame:
    """Analyzes a dynamic strategy that adjusts its daily investment to target min/max duration."""
    n_paths, n_steps = price_paths.shape
    results = []
    
    for i in range(n_paths):
        path = price_paths[i, :]
        # --- Actual (Variable Duration) Execution ---
        remaining_investment = total_investment
        total_shares = 0
        days_invested = 0
        total_cost = 0
        for t in range(1, len(path)):
            if days_invested >= max_investment_days or remaining_investment <= 1e-6: break
            hist_prices = path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else path[t]
            target_duration = min_investment_days if path[t] < running_avg else max_investment_days
            remaining_days = target_duration - days_invested
            target_investment = remaining_investment / remaining_days if remaining_days > 0 else remaining_investment
            actual_investment = min(target_investment, remaining_investment)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            total_shares += shares_bought
            total_cost += actual_investment
            remaining_investment -= actual_investment
            days_invested += 1

        avg_execution_price = total_cost / total_shares if total_shares > 0 else 0
        simple_avg_price = np.mean(path[1:days_invested+1]) if days_invested > 0 else 0
        total_performance_bps = (1 - (avg_execution_price / simple_avg_price)) * 10000 if simple_avg_price > 0 else 0

        # --- Fixed Duration Benchmark for Timing Alpha ---
        fixed_duration = int((min_investment_days + max_investment_days) / 2)
        fixed_total_cost = 0
        fixed_total_shares = 0
        remaining_investment_fixed = total_investment
        for t in range(1, fixed_duration + 1):
            if t >= len(path): break
            hist_prices = path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else path[t]
            target_duration = min_investment_days if path[t] < running_avg else max_investment_days
            remaining_days = target_duration - (t - 1)
            target_investment = remaining_investment_fixed / remaining_days if remaining_days > 0 else remaining_investment_fixed
            actual_investment = min(target_investment, remaining_investment_fixed)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            fixed_total_shares += shares_bought
            fixed_total_cost += actual_investment
            remaining_investment_fixed -= actual_investment

        fixed_avg_exec_price = fixed_total_cost / fixed_total_shares if fixed_total_shares > 0 else 0
        fixed_simple_avg_price = np.mean(path[1:fixed_duration+1]) if fixed_duration > 0 else 0
        timing_alpha_bps = (1 - (fixed_avg_exec_price / fixed_simple_avg_price)) * 10000 if fixed_simple_avg_price > 0 else 0
        duration_alpha_bps = total_performance_bps - timing_alpha_bps
        
        results.append({
            'path_id': i + 1, 'terminal_price': path[-1], 'total_usd_invested': total_cost,
            'total_shares_acquired': total_shares, 'avg_execution_price': avg_execution_price,
            'simple_avg_price': simple_avg_price, 'total_performance_bps': total_performance_bps,
            'timing_alpha_bps': timing_alpha_bps, 'duration_alpha_bps': duration_alpha_bps,
            'actual_investment_days': days_invested
        })
    return pd.DataFrame(results)

def analyze_vol_scaled_adaptive_strategy(price_paths: np.ndarray, total_investment: float, min_investment_days: int, max_investment_days: int, sigma: float) -> pd.DataFrame:
    """Analyzes a strategy that dynamically scales its target duration based on price deviation from the mean."""
    n_paths, n_steps = price_paths.shape
    results = []

    for i in range(n_paths):
        path = price_paths[i, :]
        # --- Actual (Variable Duration) Execution ---
        remaining_investment = total_investment
        total_shares = 0
        days_invested = 0
        total_cost = 0
        for t in range(1, len(path)):
            if days_invested >= max_investment_days or (remaining_investment <= 1e-6 and days_invested >= min_investment_days): break
            hist_prices = path[1:t]
            z_score = 0
            if len(hist_prices) >= 2:
                running_avg, running_std = np.mean(hist_prices), np.std(hist_prices)
                z_score = (path[t] - running_avg) / running_std if running_std > 1e-6 else 0
            
            mid_duration = (min_investment_days + max_investment_days) / 2
            duration_range = (max_investment_days - min_investment_days) / 2
            target_duration = mid_duration + np.tanh(z_score / 2) * duration_range
            remaining_days = target_duration - days_invested
            target_investment = remaining_investment / remaining_days if remaining_days > 0 else remaining_investment
            actual_investment = min(target_investment, remaining_investment)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            total_shares += shares_bought
            total_cost += actual_investment
            remaining_investment -= actual_investment
            days_invested += 1

        avg_execution_price = total_cost / total_shares if total_shares > 0 else 0
        simple_avg_price = np.mean(path[1:days_invested+1]) if days_invested > 0 else 0
        total_performance_bps = (1 - (avg_execution_price / simple_avg_price)) * 10000 if simple_avg_price > 0 else 0

        # --- Fixed Duration Benchmark for Timing Alpha ---
        fixed_duration = int(mid_duration)
        fixed_total_cost = 0
        fixed_total_shares = 0
        remaining_investment_fixed = total_investment
        for t in range(1, fixed_duration + 1):
            if t >= len(path): break
            hist_prices = path[1:t]
            z_score = 0
            if len(hist_prices) >= 2:
                running_avg, running_std = np.mean(hist_prices), np.std(hist_prices)
                z_score = (path[t] - running_avg) / running_std if running_std > 1e-6 else 0
            
            target_duration = mid_duration + np.tanh(z_score / 2) * duration_range
            remaining_days = target_duration - (t - 1)
            target_investment = remaining_investment_fixed / remaining_days if remaining_days > 0 else remaining_investment_fixed
            actual_investment = min(target_investment, remaining_investment_fixed)
            shares_bought = actual_investment / path[t] if path[t] > 0 else 0
            fixed_total_shares += shares_bought
            fixed_total_cost += actual_investment
            remaining_investment_fixed -= actual_investment

        fixed_avg_exec_price = fixed_total_cost / fixed_total_shares if fixed_total_shares > 0 else 0
        fixed_simple_avg_price = np.mean(path[1:fixed_duration+1]) if fixed_duration > 0 else 0
        timing_alpha_bps = (1 - (fixed_avg_exec_price / fixed_simple_avg_price)) * 10000 if fixed_simple_avg_price > 0 else 0
        duration_alpha_bps = total_performance_bps - timing_alpha_bps
        
        results.append({
            'path_id': i + 1, 'terminal_price': path[-1], 'total_usd_invested': total_cost,
            'total_shares_acquired': total_shares, 'avg_execution_price': avg_execution_price,
            'simple_avg_price': simple_avg_price, 'total_performance_bps': total_performance_bps,
            'timing_alpha_bps': timing_alpha_bps, 'duration_alpha_bps': duration_alpha_bps,
            'actual_investment_days': days_invested
        })
    return pd.DataFrame(results)

# --- Visualization Functions ---
def plot_performance_distribution(results_df: pd.DataFrame, col_name: str, title: str):
    fig = px.histogram(results_df, x=col_name, nbins=75, histnorm='probability density', title=title, labels={col_name: 'Performance (bps)'})
    fig.add_vline(x=0.0, line_width=2, line_dash="dash", line_color="red", annotation_text="Neutral (0 bps)", annotation_position="top right")
    return fig

def plot_investment_days_distribution(results_df: pd.DataFrame):
    fig = px.histogram(results_df, x='actual_investment_days', nbins=50, title='Distribution of Buyback Program Duration', labels={'actual_investment_days': 'Number of Days to Complete Buyback'})
    return fig

def plot_executed_usd_distribution(results_df: pd.DataFrame):
    fig = px.histogram(results_df, x='total_usd_invested', nbins=50, title='Distribution of Total Capital Deployed', labels={'total_usd_invested': 'Total USD Deployed'})
    return fig

def plot_single_path_dca_analysis(price_path: np.ndarray, daily_investment: float):
    investment_prices = price_path[1:]
    days_axis = np.arange(1, len(price_path))
    shares_daily = daily_investment / investment_prices
    cumulative_shares = np.cumsum(shares_daily)
    cumulative_investment = days_axis * daily_investment
    running_avg_exec_price = cumulative_investment / cumulative_shares
    running_simple_avg_price = pd.Series(investment_prices).expanding().mean().to_numpy()
    running_performance_bps = (1 - (running_avg_exec_price / running_simple_avg_price)) * 10000
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=np.arange(len(price_path)), y=price_path, name="Stock Price", line=dict(color='royalblue')), secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_avg_exec_price, name="Running Avg. Buyback Price", line=dict(color='green', dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_simple_avg_price, name="Running Simple Avg. Price", line=dict(color='orange', dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_performance_bps, name="Performance (bps)", line=dict(color='firebrick', dash='dot')), secondary_y=True)
    
    fig.update_layout(title_text=f'Deep Dive on Path #1 (Daily Fixed Amount Strategy)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="<b>Price (USD)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Performance (bps)</b>", secondary_y=True)
    return fig
    
def plot_single_path_variable_analysis(price_path: np.ndarray, params: dict):
    daily_investments = []
    remaining_investment = params['total_investment']
    days_invested = 0

    # Re-run logic for a single path to get daily data for plotting
    for t in range(1, len(price_path)):
        if params['name'] == 'Adaptive DCA':
            if (t-1) >= params['max_investment_days'] or (remaining_investment <= 1e-6 and (t-1) >= params['min_investment_days']): break
            base_daily_investment = params['total_investment'] / params['base_n_days']
            hist_prices = price_path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else price_path[t]
            modulation = params['modulation_factor'] if price_path[t] < running_avg else 1 / params['modulation_factor']
            actual_investment = min(base_daily_investment * modulation, remaining_investment)
        elif params['name'] == 'Dynamic Adaptive DCA':
            if (t-1) >= params['max_investment_days'] or remaining_investment <= 1e-6: break
            hist_prices = price_path[1:t]
            running_avg = np.mean(hist_prices) if len(hist_prices) > 0 else price_path[t]
            target_duration = params['min_investment_days'] if price_path[t] < running_avg else params['max_investment_days']
            remaining_days = target_duration - (t - 1)
            target_investment = remaining_investment / remaining_days if remaining_days > 0 else remaining_investment
            actual_investment = min(target_investment, remaining_investment)
        elif params['name'] == 'Volatility-Scaled Adaptive':
            if (t-1) >= params['max_investment_days'] or (remaining_investment <= 1e-6 and (t-1) >= params['min_investment_days']): break
            hist_prices = price_path[1:t]
            if len(hist_prices) < 2:
                z_score = 0
            else:
                running_avg, running_std = np.mean(hist_prices), np.std(hist_prices)
                z_score = (price_path[t] - running_avg) / running_std if running_std > 1e-6 else 0
            
            mid_duration = (params['min_investment_days'] + params['max_investment_days']) / 2
            duration_range = (params['max_investment_days'] - params['min_investment_days']) / 2
            target_duration = mid_duration + np.tanh(z_score / 2) * duration_range
            
            remaining_days = target_duration - (t - 1)
            target_investment = remaining_investment / remaining_days if remaining_days > 0 else remaining_investment
            actual_investment = min(target_investment, remaining_investment)
        
        daily_investments.append(actual_investment)
        remaining_investment -= actual_investment
        days_invested += 1

    daily_investments = np.array(daily_investments)
    
    # Truncate all data to the actual number of days invested
    days_axis = np.arange(1, days_invested + 1)
    investment_prices = price_path[1:days_invested+1]

    shares_daily = np.divide(daily_investments, investment_prices, out=np.zeros_like(daily_investments), where=investment_prices!=0)
    cumulative_shares = np.cumsum(shares_daily)
    cumulative_investment = np.cumsum(daily_investments)
    running_avg_exec_price = np.divide(cumulative_investment, cumulative_shares, out=np.zeros_like(cumulative_investment), where=cumulative_shares!=0)
    running_simple_avg_price = pd.Series(investment_prices).expanding().mean().to_numpy()
    running_performance_bps = (1 - np.divide(running_avg_exec_price, running_simple_avg_price, out=np.ones_like(running_avg_exec_price), where=running_simple_avg_price!=0)) * 10000

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{}]])
    fig.add_trace(go.Scatter(x=days_axis, y=investment_prices, name="Stock Price", line=dict(color='royalblue')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_simple_avg_price, name="Running Avg. Price", line=dict(color='orange', dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_avg_exec_price, name="Running Avg. Exec Price", line=dict(color='green', dash='dash')), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=days_axis, y=running_performance_bps, name="Performance (bps)", line=dict(color='firebrick', dash='dot')), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=days_axis, y=daily_investments, name="Daily Buyback Amount", marker_color='darkcyan'), row=2, col=1)

    fig.update_layout(title_text=f"Deep Dive on Path #1 ({params['name']})", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="<b>Price (USD)</b>", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="<b>Performance (bps)</b>", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="<b>Buyback (USD)</b>", row=2, col=1)
    return fig

def plot_performance_vs_terminal_price(results_df: pd.DataFrame):
    fig = px.scatter(results_df, x='terminal_price', y='total_performance_bps', title='Execution Performance vs. Stock Price Outcome', labels={'terminal_price': 'Terminal Stock Price (USD)', 'total_performance_bps': 'Total Performance (bps)'}, hover_data=['path_id'], color='total_performance_bps', color_continuous_scale=px.colors.sequential.Viridis)
    fig.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="red")
    return fig

def plot_price_paths(price_paths: np.ndarray, n_to_plot: int = 50):
    n_paths, n_steps = price_paths.shape
    path_ids_to_plot = np.random.choice(range(n_paths), size=min(n_paths, n_to_plot), replace=False)
    df_list = [pd.DataFrame({'Day': range(n_steps), 'Price': price_paths[path_id, :], 'Path': f'Path_{path_id+1}'}) for path_id in path_ids_to_plot]
    df_to_plot = pd.concat(df_list)
    fig = px.line(df_to_plot, x='Day', y='Price', color='Path', title=f'Sample of {n_to_plot} Simulated Stock Price Paths', labels={'Day': 'Trading Day', 'Price': 'Stock Price (USD)'})
    fig.update_layout(showlegend=False)
    return fig

# --- Caching ---
@st.cache_data
def run_full_simulation(s0, mu, sigma, n_days, n_paths, strategy_params):
    price_paths = run_gbm_simulation(s0, mu, sigma, n_days, n_paths)
    strategy = strategy_params['name']
    
    params_for_analysis = strategy_params.copy()
    del params_for_analysis['name']

    if strategy == 'Daily DCA':
        return price_paths, analyze_dca_strategy(price_paths, **params_for_analysis)
    elif strategy == 'Adaptive DCA':
        return price_paths, analyze_adaptive_strategy(price_paths, **params_for_analysis)
    elif strategy == 'Dynamic Adaptive DCA':
        if 'base_n_days' in params_for_analysis: del params_for_analysis['base_n_days']
        if 'modulation_factor' in params_for_analysis: del params_for_analysis['modulation_factor']
        return price_paths, analyze_dynamic_adaptive_strategy(price_paths, **params_for_analysis)
    elif strategy == 'Volatility-Scaled Adaptive':
        if 'base_n_days' in params_for_analysis: del params_for_analysis['base_n_days']
        if 'modulation_factor' in params_for_analysis: del params_for_analysis['modulation_factor']
        return price_paths, analyze_vol_scaled_adaptive_strategy(price_paths, **params_for_analysis)

    raise ValueError("Unknown strategy selected")

# --- Main Application UI and Logic ---
st.title("🏦 Share Buyback Pre-Trade Simulator")
st.markdown("This application uses a Monte Carlo simulation to analyze and compare different execution strategies for a corporate share repurchase program.")

with st.sidebar:
    st.header("⚙️ Market Parameters")
    s0 = st.number_input("Initial Stock Price ($S_0$)", 1.0, value=100.0, step=1.0, help="The starting stock price for all simulations.")
    mu = st.slider("Annualized Drift (μ)", -0.5, 0.5, 0.0, 0.01, "%.2f", help="The expected annualized rate of return for the stock.")
    sigma = st.slider("Annualized Volatility (σ)", 0.05, 1.0, 0.25, 0.01, "%.2f", help="The annualized standard deviation of the stock's log returns, a measure of risk and price fluctuation.")
    
    st.header("📈 Buyback Program Parameters")
    strategy_name = st.selectbox("Choose Execution Strategy", ["Daily DCA", "Adaptive DCA", "Dynamic Adaptive DCA", "Volatility-Scaled Adaptive"], help="Select the buyback execution strategy to simulate.")

    strategy_params = {'name': strategy_name}
    if strategy_name == 'Daily DCA':
        n_days = st.number_input("Buyback Period (Days)", 10, value=125, step=5, help="The fixed number of days over which to execute the buyback.")
        daily_investment = st.number_input("Daily Buyback Amount (USD)", 1.0, value=1000000.0, step=10000.0, format="%f", help="The fixed US dollar amount to spend on repurchases each day.")
        strategy_params['daily_investment'] = daily_investment
    else: # All other strategies
        n_days = st.number_input("Max Simulation Period (Days)", 10, value=300, step=5, help="The absolute maximum number of days for the simulation to run.")
        total_investment = st.number_input("Total Buyback Amount (USD)", 1000.0, value=125000000.0, step=100000.0, format="%f", help="The total capital authorized for the buyback program.")
        
        st.markdown("**Execution Duration Range**")
        col1, col2 = st.columns(2)
        min_duration = col1.number_input("Min Duration (Days)", 10, value=75, step=5, help="The strategy will aim to execute for at least this many days.")
        max_duration = col2.number_input("Max Duration (Days)", min_duration + 1, value=175, step=5, help="The strategy will not execute beyond this day, even if budget remains.")
        
        strategy_params.update({
            'total_investment': total_investment,
            'min_investment_days': min_duration,
            'max_investment_days': max_duration,
        })

        if strategy_name == 'Adaptive DCA':
            base_n_days = st.number_input("Target Duration (Days)", 10, value=125, step=5, help="The desired or 'baseline' buyback duration.")
            if min_duration >= base_n_days or max_duration <= base_n_days:
                st.error("Target Duration must be between Min and Max Duration.")
                st.stop()
            m_from_min = base_n_days / min_duration if min_duration > 0 else 1.0
            m_from_max = max_duration / base_n_days if base_n_days > 0 else 1.0
            modulation_factor = (m_from_min + m_from_max) / 2.0
            st.info(f"Implied Modulation Factor: {modulation_factor:.2f}", icon="✅")
            strategy_params.update({'base_n_days': base_n_days, 'modulation_factor': modulation_factor})
        
        if strategy_name == 'Volatility-Scaled Adaptive':
            strategy_params['sigma'] = sigma

    st.header("🎲 Monte Carlo Parameters")
    n_paths = st.number_input("Number of Simulations", 100, value=500, step=100, help="The number of independent price paths to simulate.")
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

if run_button:
    with st.spinner(f"Running {n_paths} simulations..."):
        price_paths, results_df = run_full_simulation(s0, mu, sigma, int(n_days), int(n_paths), strategy_params)
        st.session_state.price_paths = price_paths
        st.session_state.results_df = results_df
        st.session_state.strategy_params = strategy_params

if 'results_df' in st.session_state and isinstance(st.session_state.results_df, pd.DataFrame):
    st.header(f"🔍 Simulation Results: {st.session_state.strategy_params['name']}")
    results = st.session_state.results_df
    
    st.subheader("Key Performance Indicators (KPIs)")
    st.markdown("These metrics summarize the performance of the strategy across all simulated paths.")
    
    cols = st.columns(4)
    cols[0].metric("Median Total Performance", f"{results['total_performance_bps'].median():.1f} bps", help="The median of the total outperformance (Timing + Duration). Positive is better.")
    cols[1].metric("Median Timing Alpha", f"{results['timing_alpha_bps'].median():.1f} bps", help="The median value added by daily buy/sell decisions within a fixed period.")
    cols[2].metric("Median Duration Alpha", f"{results['duration_alpha_bps'].median():.1f} bps", help="The median value added by dynamically changing the program's length.")
    cols[3].metric("Favorable Outcomes", f"{(results['total_performance_bps'] > 0).mean():.1%}", help="Percentage of outcomes where the total performance was positive.")

    tab_titles = ["Summary & Distributions", "Single Path Deep Dive", "Aggregate Analysis", "Raw Data", "Explanations"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    with tab1:
        st.markdown("### Understanding the Distributions")
        c1, c2 = st.columns(2)
        c1.plotly_chart(plot_performance_distribution(results, 'total_performance_bps', 'Distribution of Total Performance'), use_container_width=True)
        if st.session_state.strategy_params['name'] != 'Daily DCA':
            c2.plotly_chart(plot_investment_days_distribution(results), use_container_width=True)
        else:
            c2.markdown("#### Terminal Stock Price Distribution")
            c2.plotly_chart(px.histogram(results, x='terminal_price', nbins=75), use_container_width=True)
        
        if st.session_state.strategy_params['name'] != 'Daily DCA':
            st.markdown("#### Performance Alpha Components")
            c3, c4 = st.columns(2)
            c3.plotly_chart(plot_performance_distribution(results, 'timing_alpha_bps', 'Distribution of Timing Alpha'), use_container_width=True)
            c4.plotly_chart(plot_performance_distribution(results, 'duration_alpha_bps', 'Distribution of Duration Alpha'), use_container_width=True)


    with tab2:
        st.markdown("### Single Path Deep Dive")
        first_path = st.session_state.price_paths[0]
        if st.session_state.strategy_params['name'] == 'Daily DCA':
            fig = plot_single_path_dca_analysis(first_path, st.session_state.strategy_params['daily_investment'])
        else:
            fig = plot_single_path_variable_analysis(first_path, st.session_state.strategy_params)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Aggregate Analysis")
        st.plotly_chart(plot_performance_vs_terminal_price(results), use_container_width=True)
        st.plotly_chart(plot_price_paths(st.session_state.price_paths, n_to_plot=100), use_container_width=True)

    with tab4:
        st.markdown("### Raw Simulation Data")
        display_cols = ['path_id', 'terminal_price', 'total_performance_bps', 'timing_alpha_bps', 'duration_alpha_bps', 'actual_investment_days', 'avg_execution_price', 'simple_avg_price']
        st.dataframe(results[display_cols].style.format(precision=2))
        
    with tab5:
        st.header("📜 Explanations and Methodology")
        
        st.subheader("The Simulation Model: Geometric Brownian Motion (GBM)")
        st.markdown(r"""
        This application models the underlying stock price movements using Geometric Brownian Motion (GBM), a standard model in quantitative finance. The price $S_t$ at any time $t$ is given by the formula:

        $$S_t = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t \right)$$

        - $S_0$: The initial stock price.
        - $\mu$ (mu): The **annualized drift**, representing the expected rate of return of the stock.
        - $\sigma$ (sigma): The **annualized volatility**, representing the magnitude of random price fluctuations.
        - $W_t$: A random component from a Wiener process, which introduces uncertainty into the price path.

        The term $(\mu - \frac{1}{2}\sigma^2)$ is the Itô-corrected drift, which is crucial for ensuring the model is mathematically sound and that the expected return of the simulated price matches the input parameter $\mu$.
        """)
        
        st.subheader("Buyback Execution Strategies")
        st.markdown("""
        **1. Daily DCA (Daily Fixed Amount):**
        - **Rule:** Repurchase a fixed dollar amount of shares every single day of the buyback period, regardless of the stock price.
        - **Characteristics:** The simplest execution strategy. It is predictable and easy to implement, but it is entirely passive and makes no attempt to react to market conditions.

        **2. Adaptive DCA:**
        - **Rule:** Repurchases a variable amount each day based on a simple rule:
            - If the current price is **below** the running average of all past prices, repurchase **more** shares than the baseline daily amount.
            - If the current price is **above** the running average, repurchase **fewer** shares.
        - **Goal:** This is a "mean-reversion" strategy. It attempts to automatically buy more shares when the stock is "cheap" relative to its recent history and less when it is "expensive."
        - **Characteristics:** An active strategy that tries to improve execution price. The total duration of the buyback is not fixed; it depends on market movements but is targeted to be within the Min/Max Duration range you set.

        **3. Dynamic Adaptive DCA:**
        - **Rule:** A more aggressive mean-reversion strategy. Each day, it recalculates its buyback size to aim for completion by either the **Min Duration** (if price is below average) or the **Max Duration** (if price is above average).
        - **Goal:** This strategy dynamically adjusts its participation rate. When it perceives favorable (low) prices, it accelerates to try and finish the entire program quickly. When it perceives unfavorable (high) prices, it slows down dramatically, preserving capital to deploy over a longer period.
        - **Characteristics:** This strategy's daily buyback amount is much more variable than the other two. It will either execute very quickly or very slowly based on market conditions.

        **4. Volatility-Scaled Adaptive:**
        - **Rule:** This is a more sophisticated strategy. It dynamically sets a target completion date based on how many standard deviations the current price is away from its running average (its "z-score").
        - **Goal:** To accelerate the buyback when the price is significantly below its recent average (a good buying opportunity) and decelerate when it is significantly above. It aims to finish the program by the `Min Duration` in the most favorable scenarios and by the `Max Duration` in the least favorable ones.
        - **Characteristics:** Its behavior is highly path-dependent and sensitive to changes in volatility, making it a powerful tool for exploring risk-adjusted execution performance.
        """)

        st.subheader("Performance Metric & Alpha Decomposition")
        st.markdown(r"""
        The key performance metric is **Execution Performance in Basis Points (bps)**. It measures how well the chosen strategy timed its repurchases compared to a naive benchmark. 
        
        #### Total Performance
        This is the overall outperformance of the strategy.
        $$\text{Total Performance (bps)} = \left( 1 - \frac{\text{Average Buyback Price}}{\text{Simple Average Stock Price}} \right) \times 10,000$$
        
        - **Average Buyback Price:** The effective price the company paid per share over the entire program (Total $ Spent / Total Shares Repurchased).
        - **Simple Average Stock Price:** The simple arithmetic mean of the stock's price over the *actual* investment period for that specific path.
        
        For adaptive strategies, we decompose this total performance into two sources of "alpha" or value-add:

        #### 1. Timing Alpha
        This measures the value generated purely by the strategy's daily decisions to buy more or less, assuming the program had a **fixed duration**. This isolates the benefit of the intra-period timing.

        #### 2. Duration Alpha
        This measures the additional value generated by the strategy's ability to **dynamically change the length** of the buyback program. It is the extra performance gained by finishing early in favorable markets or extending the program in unfavorable ones.
        
        $$\text{Total Performance} = \text{Timing Alpha} + \text{Duration Alpha}$$
        
        By breaking down the performance this way, we can understand whether a strategy's edge comes from its daily trading logic or from its higher-level decision to change the program's overall timeline.
        """)

else:
    st.info("Please configure parameters and run the simulation.")
