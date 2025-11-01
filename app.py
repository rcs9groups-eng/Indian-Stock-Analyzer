import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import ta

# --- STREAMLIT CLOUD OPTIMIZED VERSION ---

# Remove authentication for easier cloud deployment
# Or use environment variables for password

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
        self.all_symbols = {
            'NIFTY 50': '^NSEI', 'BANK NIFTY': '^NSEBANK', 'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS', 'ICICI BANK': 'ICICIBANK.NS', 'SBI': 'SBIN.NS', 'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'LT': 'LT.NS', 'ITC': 'ITC.NS', 'HUL': 'HINDUNILVR.NS', 'ASIAN PAINTS': 'ASIANPAINT.NS', 'MARUTI': 'MARUTI.NS',
            'TITAN': 'TITAN.NS', 'SUN PHARMA': 'SUNPHARMA.NS', 'AXIS BANK': 'AXISBANK.NS', 'KOTAK BANK': 'KOTAKBANK.NS',
            'BAJFINANCE': 'BAJFINANCE.NS', 'WIPRO': 'WIPRO.NS', 'HCL TECH': 'HCLTECH.NS', 'ULTRACEMCO': 'ULTRACEMCO.NS',
            'NESTLE': 'NESTLEIND.NS', 'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS', 'ONGC': 'ONGC.NS', 'M&M': 'M&M.NS',
            'TATA MOTORS': 'TATAMOTORS.NS', 'ADANI ENTERPRISES': 'ADANIENT.NS'
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
        
        st.subheader("🎯 ANALYSIS PERIOD")
        period = st.selectbox("Data Period", ["6mo", "1y", "2y"], index=1)
        
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

    # Sidebar
    st.sidebar.header("🔧 SETTINGS")
    st.sidebar.info("Configure your analysis parameters")
    
    st.sidebar.header("📈 QUICK ACTIONS")
    if st.sidebar.button("🔄 Clear Cache"):
        st.rerun()
    
    st.sidebar.header("ℹ️ ABOUT")
    st.sidebar.info("""
    This tool provides technical analysis using:
    - 20+ Technical Indicators
    - AI-powered Scoring
    - Real-time Market Data
    - Advanced Charting
    """)

if __name__ == "__main__":
    main()
