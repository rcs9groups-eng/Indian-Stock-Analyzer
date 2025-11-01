import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import ta

# --- STREAMLIT CLOUD OPTIMIZED VERSION - NO CUSTOM SYMBOLS ---

st.set_page_config(
    page_title="ULTRA STOCK ANALYZER PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2563eb; text-align: center; margin-bottom: 1rem; font-weight: bold; background: linear-gradient(45deg, #2563eb, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .private-badge { background: #ef4444; color: white; padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.8rem; margin-left: 1rem; }
    .super-card { background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 5px solid; border: 1px solid #e5e7eb; }
    .ultra-buy { border-left-color: #10b981; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
    .strong-buy { border-left-color: #22c55e; background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%); }
    .strong-sell { border-left-color: #ef4444; background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); }
    .hold { border-left-color: #f59e0b; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); }
    .indicator-box { background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #e5e7eb; margin: 0.5rem 0; }
    .bullish { border-color: #10b981; background: #d1fae5; }
    .bearish { border-color: #ef4444; background: #fee2e2; }
    .neutral { border-color: #f59e0b; background: #fef3c7; }
    .trade-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; margin: 1rem 0; }
    .calculator-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1rem; border-radius: 12px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzerPro:
    def __init__(self):
        # Predefined symbols only - no custom input
        self.all_symbols = {
            'NIFTY 50': '^NSEI', 
            'BANK NIFTY': '^NSEBANK', 
            'RELIANCE': 'RELIANCE.NS', 
            'TCS': 'TCS.NS', 
            'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS', 
            'ICICI BANK': 'ICICIBANK.NS', 
            'SBI': 'SBIN.NS', 
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS', 
            'ITC': 'ITC.NS', 
            'HUL': 'HINDUNILVR.NS', 
            'ASIAN PAINTS': 'ASIANPAINT.NS', 
            'MARUTI': 'MARUTI.NS',
            'TITAN': 'TITAN.NS', 
            'SUN PHARMA': 'SUNPHARMA.NS', 
            'AXIS BANK': 'AXISBANK.NS', 
            'KOTAK BANK': 'KOTAKBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 
            'WIPRO': 'WIPRO.NS', 
            'HCL TECH': 'HCLTECH.NS', 
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'NESTLE': 'NESTLEIND.NS', 
            'POWERGRID': 'POWERGRID.NS', 
            'NTPC': 'NTPC.NS', 
            'ONGC': 'ONGC.NS', 
            'M&M': 'M&M.NS',
            'TATA MOTORS': 'TATAMOTORS.NS', 
            'ADANI ENTERPRISES': 'ADANIENT.NS',
            'TECH MAHINDRA': 'TECHM.NS',
            'WIPRO': 'WIPRO.NS',
            'HINDALCO': 'HINDALCO.NS',
            'JSW STEEL': 'JSWSTEEL.NS'
        }
    
    def get_stock_data(self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                st.error(f"❌ No data found for {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            return None

    def calculate_advanced_indicators(self, data):
        df = data.copy()
        
        try:
            # Trend Indicators
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Other indicators
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Volatility
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Support & Resistance
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            
        except Exception as e:
            st.warning(f"Some indicators couldn't be calculated: {e}")
        
        return df

    def calculate_position_size(self, capital, risk_per_trade, entry_price, stop_loss):
        risk_amount = capital * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        if price_risk > 0:
            shares = risk_amount / price_risk
            return int(shares)
        return 0

    def calculate_ai_score(self, df):
        if len(df) < 50:
            return [50] * len(df)
        
        scores = []
        for i in range(len(df)):
            if i < 50:
                scores.append(50)
                continue
                
            current = df.iloc[i]
            score = 50
            
            try:
                # Price vs Moving Averages
                if not pd.isna(current['SMA_20']) and current['Close'] > current['SMA_20']:
                    score += 10
                if not pd.isna(current['SMA_50']) and current['Close'] > current['SMA_50']:
                    score += 10
                
                # MACD Signal
                if not pd.isna(current['MACD']) and not pd.isna(current['MACD_Signal']) and current['MACD'] > current['MACD_Signal']:
                    score += 7.5
                
                # RSI Analysis
                if not pd.isna(current['RSI']):
                    if 30 < current['RSI'] < 70:
                        score += 7.5
                    elif current['RSI'] > 50:
                        score += 5
                
                # Stochastic
                if not pd.isna(current['Stoch_K']) and not pd.isna(current['Stoch_D']):
                    if current['Stoch_K'] > current['Stoch_D'] and current['Stoch_K'] < 80:
                        score += 5
                
                # ADX Trend Strength
                if not pd.isna(current['ADX']) and current['ADX'] > 25:
                    score += 5
                
                # Volume Analysis
                if i > 0 and not pd.isna(current['OBV']) and not pd.isna(df['OBV'].iloc[i-1]) and current['OBV'] > df['OBV'].iloc[i-1]:
                    score += 5
                
                # Support/Resistance
                if not pd.isna(current['Support']) and current['Close'] > current['Support']:
                    score += 5
                    
            except Exception:
                pass
            
            scores.append(min(100, max(0, score)))
        
        return scores

    def get_trading_signal(self, score):
        if score >= 80:
            return "ULTRA BUY", "ultra-buy"
        elif score >= 70:
            return "STRONG BUY", "strong-buy"
        elif score >= 60:
            return "BUY", "bullish"
        elif score >= 40:
            return "HOLD", "hold"
        elif score >= 30:
            return "SELL", "bearish"
        else:
            return "STRONG SELL", "strong-sell"

    def create_advanced_chart(self, df, symbol):
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Chart', 'MACD', 'RSI', 'Volume'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Price Chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add moving averages if available
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
        
        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Volume
        colors = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=4, col=1)
        
        fig.update_layout(height=600, title_text=f"Technical Analysis - {symbol}", template="plotly_dark")
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

    # --- BACKTESTING ENGINE ---
    def run_backtest(self, data, initial_capital=100000, buy_score=75, sell_score=50, stop_loss_pct=8.0, target_pct=15.0):
        st.write("### 🔬 Running Historical Backtest...")
        
        backtest_df = data.copy()
        
        # Generate historical signals
        scores = []
        for i in range(len(backtest_df)):
            if i < 50:
                scores.append(50)
            else:
                window_df = backtest_df.iloc[:i+1]
                window_df = self.calculate_advanced_indicators(window_df)
                score = self.calculate_ai_score(window_df)[-1]
                scores.append(score)
        
        backtest_df['score'] = scores
        backtest_df['signal'] = backtest_df['score'].apply(lambda s: 'BUY' if s >= buy_score else ('SELL' if s < sell_score else 'HOLD'))

        # Simulate Trading
        capital = initial_capital
        position = 0  # 0 for no position, 1 for long
        entry_price = 0
        trades = []
        portfolio_values = [initial_capital] * len(backtest_df)

        for i in range(1, len(backtest_df)):
            current_row = backtest_df.iloc[i]
            prev_signal = backtest_df.iloc[i-1]['signal']
            
            portfolio_values[i] = portfolio_values[i-1]  # Default: carry previous value

            if position == 0:  # Not in a position
                if prev_signal == 'BUY':
                    position = 1
                    entry_price = current_row['Open']
                    trades.append({'entry_date': current_row.name, 'entry_price': entry_price, 'type': 'LONG'})
            
            elif position == 1:  # In a long position
                stop_loss_price = entry_price * (1 - stop_loss_pct/100)
                target_price = entry_price * (1 + target_pct/100)
                
                # Check for exit conditions
                exit_signal = False
                exit_price = current_row['Open']
                exit_reason = "Signal"
                
                if current_row['Low'] <= stop_loss_price:
                    exit_signal = True
                    exit_price = stop_loss_price
                    exit_reason = "Stop Loss"
                elif current_row['High'] >= target_price:
                    exit_signal = True
                    exit_price = target_price
                    exit_reason = "Target"
                elif prev_signal == 'SELL':
                    exit_signal = True
                    exit_reason = "Sell Signal"
                
                if exit_signal:
                    pnl_ratio = (exit_price - entry_price) / entry_price
                    capital = capital * (1 + pnl_ratio)
                    portfolio_values[i] = capital
                    
                    trades[-1].update({
                        'exit_date': current_row.name, 
                        'exit_price': exit_price, 
                        'pnl': pnl_ratio,
                        'exit_reason': exit_reason
                    })
                    position = 0

        # Add portfolio values to dataframe
        backtest_df['portfolio_value'] = portfolio_values

        # Final Calculations
        if not trades:
            return None, None, None

        trades_df = pd.DataFrame(trades)
        total_return = (capital / initial_capital) - 1
        
        if 'pnl' in trades_df.columns:
            trades_df.dropna(subset=['pnl'], inplace=True)
            if len(trades_df) > 0:
                win_rate = (trades_df['pnl'] > 0).mean()
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
                avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
                
                # Sharpe Ratio
                daily_returns = backtest_df['portfolio_value'].pct_change().dropna()
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
                
                # Max Drawdown
                roll_max = backtest_df['portfolio_value'].cummax()
                daily_drawdown = backtest_df['portfolio_value'] / roll_max - 1.0
                max_drawdown = daily_drawdown.min()

                results = {
                    "Total Return": total_return, 
                    "Win Rate": win_rate, 
                    "Sharpe Ratio": sharpe_ratio, 
                    "Max Drawdown": max_drawdown,
                    "Total Trades": len(trades_df), 
                    "Avg Win %": avg_win, 
                    "Avg Loss %": avg_loss,
                    "Final Capital": capital
                }
                return results, backtest_df, trades_df
        
        return None, None, None

def main():
    app = UltraStockAnalyzerPro()
    
    st.markdown('<h1 class="main-header">🚀 ULTRA STOCK ANALYZER PRO</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 SELECT STOCK")
        selected_stock = st.selectbox("Choose Stock:", list(app.all_symbols.keys()))
        symbol = app.all_symbols[selected_stock]
        
        st.subheader("⚙️ TRADING SETTINGS")
        capital = st.number_input("Capital (₹)", min_value=1000, value=100000, step=5000)
        risk_per_trade = st.slider("Risk per Trade %", 1.0, 10.0, 2.0, 0.5)
        stop_loss_percent = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5)
        target_percent = st.slider("Target %", 1.0, 50.0, 15.0, 1.0)
        
        st.subheader("🎯 ANALYSIS PERIOD")
        period = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1)
        
        st.subheader("🧮 POSITION CALCULATOR")
        entry_price_calc = st.number_input("Entry Price", min_value=1.0, value=100.0, step=1.0)
        stop_loss_calc = st.number_input("Stop Loss Price", min_value=0.1, value=92.0, step=1.0)
        
        if st.button("Calculate Position Size", key="calc_pos"):
            shares = app.calculate_position_size(capital, risk_per_trade, entry_price_calc, stop_loss_calc)
            investment = shares * entry_price_calc
            risk_amount = capital * (risk_per_trade / 100)
            
            st.markdown(f"""
            <div class="calculator-box">
                <h3>📊 Position Sizing</h3>
                <p>Shares: <strong>{shares}</strong></p>
                <p>Investment: <strong>₹{investment:,.0f}</strong></p>
                <p>Risk Amount: <strong>₹{risk_amount:,.0f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 ANALYZE NOW", type="primary", use_container_width=True):
            with st.spinner("Analyzing stock data..."):
                data = app.get_stock_data(symbol, period)
                if data is not None and not data.empty:
                    df = app.calculate_advanced_indicators(data)
                    scores = app.calculate_ai_score(df)
                    
                    if scores:
                        current_score = scores[-1]
                        signal, signal_class = app.get_trading_signal(current_score)
                        current_price = df['Close'].iloc[-1]
                        
                        # Display Signal
                        st.markdown(f"""
                        <div class="super-card {signal_class}">
                            <h2>🎯 {signal}</h2>
                            <h3>AI Score: {current_score:.1f}/100</h3>
                            <p>Current Price: ₹{current_price:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key Metrics
                        cols = st.columns(4)
                        metrics = [
                            ('RSI', 'RSI', 'bullish' if 'RSI' in df.columns and 30 < df['RSI'].iloc[-1] < 70 else 'bearish'),
                            ('MACD', 'MACD', 'bullish' if 'MACD' in df.columns and 'MACD_Signal' in df.columns and df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'),
                            ('TREND', 'BULLISH' if 'SMA_20' in df.columns and df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'BEARISH', 'bullish' if 'SMA_20' in df.columns and df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else 'bearish'),
                            ('VOLUME', 'ACCUMULATION' if 'OBV' in df.columns and df['OBV'].iloc[-1] > df['OBV'].iloc[-2] if len(df) > 1 else True else 'DISTRIBUTION', 'bullish' if 'OBV' in df.columns and df['OBV'].iloc[-1] > df['OBV'].iloc[-2] if len(df) > 1 else True else 'bearish')
                        ]
                        
                        for col, (name, value, color) in zip(cols, metrics):
                            with col:
                                if name == 'RSI' and 'RSI' in df.columns:
                                    value = f"{df['RSI'].iloc[-1]:.1f}"
                                elif name == 'MACD' and 'MACD' in df.columns:
                                    value = f"{df['MACD'].iloc[-1]:.3f}"
                                
                                st.markdown(f'<div class="indicator-box {color}"><h4>{name}</h4><h3>{value}</h3></div>', unsafe_allow_html=True)
                        
                        # Trading Recommendations
                        shares = app.calculate_position_size(capital, risk_per_trade, current_price, current_price * (1 - stop_loss_percent/100))
                        
                        st.markdown(f"""
                        <div class="trade-box">
                            <h3>💎 TRADE RECOMMENDATION</h3>
                            <p><strong>Entry:</strong> ₹{current_price:.2f}</p>
                            <p><strong>Stop Loss:</strong> ₹{current_price * (1 - stop_loss_percent/100):.2f} ({stop_loss_percent}%)</p>
                            <p><strong>Target:</strong> ₹{current_price * (1 + target_percent/100):.2f} ({target_percent}%)</p>
                            <p><strong>Position Size:</strong> {shares} shares (₹{shares * current_price:,.0f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Chart
                        st.plotly_chart(app.create_advanced_chart(df.tail(100), selected_stock), use_container_width=True)
                        
                        # Score History
                        st.subheader("📈 SCORE HISTORY")
                        score_chart = pd.DataFrame({
                            'Date': df.index,
                            'Score': scores
                        }).tail(50)
                        
                        fig_score = go.Figure()
                        fig_score.add_trace(go.Scatter(
                            x=score_chart['Date'], 
                            y=score_chart['Score'],
                            fill='tozeroy',
                            line=dict(color='blue')
                        ))
                        fig_score.add_hline(y=70, line_dash="dash", line_color="green")
                        fig_score.add_hline(y=30, line_dash="dash", line_color="red")
                        fig_score.update_layout(title="AI Trading Score", height=300)
                        st.plotly_chart(fig_score, use_container_width=True)

    # Sidebar - Backtesting
    st.sidebar.header("🔬 BACKTESTING")
    st.sidebar.info("Test the AI strategy on historical data.")
    backtest_buy_score = st.sidebar.slider("Buy Score Threshold", 50, 100, 75)
    backtest_sell_score = st.sidebar.slider("Sell Score Threshold", 0, 50, 45)
    backtest_capital = st.sidebar.number_input("Backtest Capital", min_value=1000, value=100000, step=5000)
    
    if st.sidebar.button("⚙️ Run Historical Backtest"):
        with st.spinner(f"🔬 Backtesting {selected_stock} strategy... This may take a moment."):
            data = app.get_stock_data(symbol, "2y")
            if data is not None and not data.empty:
                results, backtest_df, trades_df = app.run_backtest(
                    data,
                    initial_capital=backtest_capital,
                    buy_score=backtest_buy_score,
                    sell_score=backtest_sell_score,
                    stop_loss_pct=stop_loss_percent,
                    target_pct=target_percent
                )

                if results:
                    st.subheader(f"🔬 Backtest Results for {selected_stock}")
                    
                    res_cols = st.columns(4)
                    res_cols[0].metric("Total Return", f"{results['Total Return']:.2%}")
                    res_cols[1].metric("Win Rate", f"{results['Win Rate']:.2%}")
                    res_cols[2].metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                    res_cols[3].metric("Max Drawdown", f"{results['Max Drawdown']:.2%}")
                    
                    res_cols2 = st.columns(3)
                    res_cols2[0].metric("Total Trades", results['Total Trades'])
                    res_cols2[1].metric("Avg Win %", f"{results['Avg Win %']:.2%}")
                    res_cols2[2].metric("Avg Loss %", f"{results['Avg Loss %']:.2%}")
                    
                    # Plot Equity Curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['portfolio_value'], 
                                           name='Strategy Equity', line=dict(color='green')))
                    # Add Buy & Hold for comparison
                    buy_hold_returns = (backtest_df['Close'] / backtest_df['Close'].iloc[0])
                    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_capital*buy_hold_returns, 
                                           name='Buy & Hold', line=dict(color='blue', dash='dot')))
                    fig.update_layout(
                        title='Portfolio Value Over Time (Equity Curve)', 
                        template='plotly_dark',
                        yaxis_title='Portfolio Value (₹)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trades table
                    if trades_df is not None and len(trades_df) > 0:
                        st.subheader("📋 Trade History")
                        trades_display = trades_df.copy()
                        trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"{x:.2%}")
                        trades_display['entry_price'] = trades_display['entry_price'].apply(lambda x: f"₹{x:.2f}")
                        trades_display['exit_price'] = trades_display['exit_price'].apply(lambda x: f"₹{x:.2f}")
                        st.dataframe(trades_display, use_container_width=True)

                else:
                    st.error("Backtest could not be completed. Not enough trades were generated.")

    # Sidebar Info
    st.sidebar.header("ℹ️ ABOUT")
    st.sidebar.info("""
    This tool provides technical analysis using:
    - 20+ Technical Indicators
    - AI-powered Scoring
    - Real-time Market Data
    - Advanced Charting
    - Historical Backtesting
    """)
    
    st.sidebar.header("🔄 QUICK ACTIONS")
    if st.sidebar.button("Clear Cache & Refresh"):
        st.rerun()

if __name__ == "__main__":
    main()
