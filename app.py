import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# --- STREAMLIT CLOUD OPTIMIZED VERSION - NO EXTERNAL TA LIBRARY ---

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

    def calculate_sma(self, data, window):
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, window):
        return data.ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd, signal)
        return macd, macd_signal

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        sma = self.calculate_sma(data, window)
        rolling_std = data.rolling(window=window).std()
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        return upper_band, lower_band

    def calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        stoch_d = stoch_k.rolling(window=d_window).mean()
        return stoch_k, stoch_d

    def calculate_obv(self, close, volume):
        obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
        return obv

    def calculate_advanced_indicators(self, data):
        df = data.copy()
        
        try:
            # Trend Indicators
            df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            df['EMA_12'] = self.calculate_ema(df['Close'], 12)
            df['EMA_26'] = self.calculate_ema(df['Close'], 26)
            
            # MACD
            df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df['High'], df['Low'], df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
            
            # Volume Indicators
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
            
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
                        
                        # RSI Metric
                        with cols[0]:
                            rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
                            rsi_color = "bullish" if 30 < rsi_value < 70 else "bearish"
                            st.markdown(f'<div class="indicator-box {rsi_color}"><h4>RSI</h4><h3>{rsi_value:.1f}</h3></div>', unsafe_allow_html=True)
                        
                        # MACD Metric
                        with cols[1]:
                            if 'MACD' in df.columns and 'MACD_Signal' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
                                macd_value = df['MACD'].iloc[-1]
                                macd_signal = "bullish" if macd_value > df['MACD_Signal'].iloc[-1] else "bearish"
                            else:
                                macd_value = 0
                                macd_signal = "neutral"
                            st.markdown(f'<div class="indicator-box {macd_signal}"><h4>MACD</h4><h3>{macd_value:.3f}</h3></div>', unsafe_allow_html=True)
                        
                        # Trend Metric
                        with cols[2]:
                            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]):
                                trend_value = "BULLISH" if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else "BEARISH"
                                trend_color = "bullish" if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else "bearish"
                            else:
                                trend_value = "NEUTRAL"
                                trend_color = "neutral"
                            st.markdown(f'<div class="indicator-box {trend_color}"><h4>TREND</h4><h3>{trend_value}</h3></div>', unsafe_allow_html=True)
                        
                        # Volume Metric
                        with cols[3]:
                            if 'OBV' in df.columns and len(df) > 1 and not pd.isna(df['OBV'].iloc[-1]):
                                volume_trend = "ACCUMULATION" if df['OBV'].iloc[-1] > df['OBV'].iloc[-2] else "DISTRIBUTION"
                                volume_color = "bullish" if df['OBV'].iloc[-1] > df['OBV'].iloc[-2] else "bearish"
                            else:
                                volume_trend = "NEUTRAL"
                                volume_color = "neutral"
                            st.markdown(f'<div class="indicator-box {volume_color}"><h4>VOLUME</h4><h3>{volume_trend}</h3></div>', unsafe_allow_html=True)
                        
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

    # Sidebar Info
    st.sidebar.header("ℹ️ ABOUT")
    st.sidebar.info("""
    This tool provides technical analysis using:
    - Moving Averages (SMA, EMA)
    - MACD
    - RSI
    - Stochastic Oscillator
    - Bollinger Bands
    - Volume Analysis (OBV)
    - AI-powered Scoring System
    """)
    
    st.sidebar.header("🔄 QUICK ACTIONS")
    if st.sidebar.button("Clear Cache & Refresh"):
        st.rerun()

if __name__ == "__main__":
    main()
