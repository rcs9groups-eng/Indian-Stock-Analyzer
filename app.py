import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Password Protection
def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.set_page_config(page_title="Login", layout="centered")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h1>🔒</h1>
                <h2>PRIVATE STOCK ANALYZER</h2>
                <p>Enter password to continue</p>
            </div>
            """, unsafe_allow_html=True)
            
            password = st.text_input("Password:", type="password", key="pw_input")
            
            if st.button("🔑 Login", use_container_width=True):
                if password == "StockMaster2024":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            
            st.stop()
    
    return True

# Authenticate user
authenticate()

# If authenticated, show the main app
st.set_page_config(
    page_title="MY PRIVATE ULTRA STOCK ANALYZER PRO",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .private-badge {
        background: #ef4444;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
    .super-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        margin: 1.5rem 0;
        border-left: 6px solid;
        border-right: 2px solid #e5e7eb;
        border-top: 2px solid #e5e7eb;
        border-bottom: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .super-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
    }
    .ultra-buy {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 50%, #6ee7b7 100%);
    }
    .strong-buy {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
    }
    .strong-sell {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 50%, #f87171 100%);
    }
    .hold {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }
    .indicator-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
        transition: transform 0.2s ease;
    }
    .indicator-box:hover {
        transform: scale(1.05);
    }
    .bullish { border-color: #10b981; background: #d1fae5; }
    .bearish { border-color: #ef4444; background: #fee2e2; }
    .neutral { border-color: #f59e0b; background: #fef3c7; }
    .trade-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .calculator-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UltraStockAnalyzerPro:
    def __init__(self):
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
            'M&M': 'M&M.NS'
        }
    
    def get_stock_data(self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return None

    def calculate_advanced_indicators(self, data):
        if data is None or len(data) < 50:
            return None
            
        df = data.copy()
        
        # 1. ENHANCED PRICE INDICATORS
        for period in [5, 8, 13, 21, 34, 50, 89, 144, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            weights = np.arange(1, period + 1)
            df[f'WMA_{period}'] = df['Close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # 2. ENHANCED MOMENTUM INDICATORS
        def calculate_rsi_enhanced(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df['RSI_14'] = calculate_rsi_enhanced(df['Close'], 14)
        df['RSI_21'] = calculate_rsi_enhanced(df['Close'], 21)
        
        ema1 = df['Close'].ewm(span=5).mean()
        ema2 = ema1.ewm(span=8).mean() 
        ema3 = ema2.ewm(span=13).mean()
        df['TEMA'] = (3 * ema1) - (3 * ema2) + ema3
        
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 3. VOLUME-BASED INDICATORS
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_RSI'] = calculate_rsi_enhanced(df['Volume'], 14)
        
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=21).mean()
        
        df['VPT'] = (df['Volume'] * (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).cumsum()
        
        # 4. ADVANCED VOLATILITY INDICATORS
        for dev in [1.5, 2, 2.5]:
            df[f'BB_Middle_{dev}'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df[f'BB_Upper_{dev}'] = df[f'BB_Middle_{dev}'] + (bb_std * dev)
            df[f'BB_Lower_{dev}'] = df[f'BB_Middle_{dev}'] - (bb_std * dev)
        
        df['BB_Width'] = (df['BB_Upper_2'] - df['BB_Lower_2']) / df['BB_Middle_2']
        df['BB_Position'] = (df['Close'] - df['BB_Lower_2']) / (df['BB_Upper_2'] - df['BB_Lower_2'])
        
        df['KC_Middle'] = df['EMA_20']
        
        # 5. SUPPORT/RESISTANCE INDICATORS
        df['Support_1'] = df['Low'].rolling(window=20).min()
        df['Resistance_1'] = df['High'].rolling(window=20).max()
        
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = (2 * df['Pivot']) - df['Low']
        df['S1'] = (2 * df['Pivot']) - df['High']
        
        # 6. TREND STRENGTH INDICATORS
        def calculate_adx(high, low, close, window=14):
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = true_range.rolling(window=window).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
            
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            adx = dx.rolling(window=window).mean()
            
            return adx, plus_di, minus_di
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        df['ADX'], df['ADX_Pos'], df['ADX_Neg'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # 7. MONEY FLOW INDICATORS
        def calculate_mfi(high, low, close, volume, window=14):
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = ((typical_price > typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            negative_flow = ((typical_price < typical_price.shift(1)) * money_flow).rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + positive_flow / negative_flow))
            return mfi
        
        df['MFI'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        def calculate_cmf(high, low, close, volume, window=20):
            mfm = ((close - low) - (high - close)) / (high - low)
            mfv = mfm * volume
            return mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
        
        df['CMF'] = calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df

    def calculate_position_size(self, capital, risk_per_trade, entry_price, stop_loss):
        risk_amount = capital * (risk_per_trade / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share > 0:
            shares = risk_amount / risk_per_share
            position_value = shares * entry_price
            return int(shares), position_value
        return 0, 0

    def calculate_ai_score(self, df):
        if df is None:
            return 0, [], {}, 0, 0
            
        current_price = df['Close'].iloc[-1]
        score = 0
        reasons = []
        signals = {}
        
        # 1. ENHANCED TREND ANALYSIS (35 points)
        trend_score = 0
        bullish_ma_count = 0
        
        fib_periods = [5, 8, 13, 21, 34, 55, 89, 144]
        for period in fib_periods:
            if f'EMA_{period}' in df and not pd.isna(df[f'EMA_{period}'].iloc[-1]):
                if current_price > df[f'EMA_{period}'].iloc[-1]:
                    trend_score += 3
                    bullish_ma_count += 1
        
        if bullish_ma_count >= 5:
            trend_score += 15
            reasons.append(f"🚀 STRONG FIBONACCI TREND ({bullish_ma_count}/8 EMAs bullish)")
            signals['Fibonacci_Trend'] = 'STRONG_BULLISH'
        
        if 'TEMA' in df and not pd.isna(df['TEMA'].iloc[-1]):
            if current_price > df['TEMA'].iloc[-1]:
                trend_score += 8
                reasons.append("📈 TEMA BULLISH - Strong trend confirmation")
                signals['TEMA'] = 'BULLISH'
        
        score += trend_score
        
        # 2. ENHANCED MOMENTUM (40 points)
        momentum_score = 0
        
        rsi_bullish = 0
        for period in [14, 21]:
            if f'RSI_{period}' in df and not pd.isna(df[f'RSI_{period}'].iloc[-1]):
                rsi_val = df[f'RSI_{period}'].iloc[-1]
                if 40 <= rsi_val <= 65:
                    momentum_score += 8
                    rsi_bullish += 1
                    reasons.append(f"🎯 Perfect RSI {period}: {rsi_val:.1f}")
                elif rsi_val < 35:
                    momentum_score += 10
                    reasons.append(f"📈 Oversold RSI {period}: {rsi_val:.1f}")
        
        if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            if (df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and 
                df['MACD_Histogram'].iloc[-1] > df['MACD_Histogram'].iloc[-2] and
                df['MACD_Histogram'].iloc[-1] > 0):
                momentum_score += 12
                reasons.append("💪 MACD STRONG BULLISH - Rising histogram")
                signals['MACD'] = 'STRONG_BULLISH'
        
        score += momentum_score
        
        # 3. VOLUME CONFIRMATION (25 points)
        volume_score = 0
        
        if all(col in df for col in ['Volume', 'Volume_SMA_20']):
            volume_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
            if volume_ratio > 2.5:
                volume_score += 15
                reasons.append("💰 VOLUME EXPLOSION - Very strong interest")
                signals['Volume'] = 'EXTREME_BULLISH'
            elif volume_ratio > 1.8:
                volume_score += 10
                reasons.append("💰 HIGH VOLUME - Strong participation")
                signals['Volume'] = 'STRONG_BULLISH'
        
        if 'OBV_EMA' in df and len(df) > 21:
            if df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1]:
                volume_score += 6
                reasons.append("📊 OBV above EMA - Accumulation")
                signals['OBV'] = 'BULLISH'
        
        if 'MFI' in df and 20 <= df['MFI'].iloc[-1] <= 80:
            volume_score += 4
            signals['MFI'] = 'BULLISH'
        
        score += volume_score
        
        # 4. VOLATILITY & RISK (20 points)
        risk_score = 0
        
        if 'BB_Position' in df and not pd.isna(df['BB_Position'].iloc[-1]):
            bb_pos = df['BB_Position'].iloc[-1]
            if 0.25 <= bb_pos <= 0.75:
                risk_score += 8
                reasons.append("📈 Ideal BB position")
                signals['Bollinger_Bands'] = 'STRONG_BULLISH'
            elif bb_pos < 0.25:
                risk_score += 12
                reasons.append("🎯 Near BB lower band - Excellent entry")
                signals['Bollinger_Bands'] = 'OVERSOLD_BULLISH'
        
        if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]):
            atr_percent = (df['ATR'].iloc[-1] / current_price) * 100
            if atr_percent < 1.8 and df['ADX'].iloc[-1] > 20:
                risk_score += 8
                reasons.append("🛡️ Low volatility with strong trend")
                signals['ATR'] = 'LOW_VOL_TREND'
        
        score += risk_score
        
        # 5. TREND STRENGTH (15 points)
        trend_strength_score = 0
        
        if 'ADX' in df and not pd.isna(df['ADX'].iloc[-1]):
            if df['ADX'].iloc[-1] > 25:
                trend_strength_score += 10
                reasons.append(f"💪 Strong trend (ADX: {df['ADX'].iloc[-1]:.1f})")
                signals['ADX'] = 'STRONG_TREND'
            elif df['ADX'].iloc[-1] > 20:
                trend_strength_score += 6
                signals['ADX'] = 'MODERATE_TREND'
        
        if all(col in df for col in ['ADX_Pos', 'ADX_Neg']):
            if df['ADX_Pos'].iloc[-1] > df['ADX_Neg'].iloc[-1]:
                trend_strength_score += 5
                signals['DI'] = 'BULLISH'
        
        score += trend_strength_score
        
        bullish_signals = sum(1 for s in signals.values() if 'BULLISH' in str(s))
        total_signals = len(signals)
        
        if bullish_signals >= 8:
            score = min(score + 20, 100)
            reasons.append(f"🚀 EXTREME BULLISH ({bullish_signals}/{total_signals} signals)")
        elif bullish_signals >= 6:
            score = min(score + 15, 100)
            reasons.append(f"📈 STRONG BULLISH ({bullish_signals}/{total_signals} signals)")
        
        return min(score, 100), reasons, signals, bullish_signals, total_signals

    def get_trading_signal(self, score):
        if score >= 95:
            return "🚀 QUANTUM STRONG BUY", "ultra-buy", "#059669", "IMMEDIATE BUY - EXTREME CONFIDENCE"
        elif score >= 90:
            return "🎯 ULTRA STRONG BUY", "ultra-buy", "#10b981", "ULTRA STRONG BUY - MAXIMUM CONFIDENCE"
        elif score >= 85:
            return "💎 VERY STRONG BUY", "ultra-buy", "#22c55e", "VERY STRONG BUY - HIGH CONFIDENCE"
        elif score >= 80:
            return "📈 POWERFUL BUY", "strong-buy", "#4ade80", "POWERFUL BUY - STRONG OPPORTUNITY"
        elif score >= 75:
            return "⚡ STRONG BUY", "strong-buy", "#84cc16", "STRONG BUY - GOOD ENTRY"
        elif score >= 70:
            return "💰 ACCUMULATE", "hold", "#f59e0b", "ACCUMULATE ON DIPS"
        elif score >= 65:
            return "🔄 HOLD", "hold", "#f97316", "HOLD - WAIT FOR CONFIRMATION"
        elif score >= 60:
            return "🔔 CAUTION", "hold", "#fb923c", "CAUTION - MONITOR CLOSELY"
        elif score >= 55:
            return "📉 REDUCE", "strong-sell", "#ef4444", "REDUCE POSITION"
        elif score >= 50:
            return "⚠️ SELL", "strong-sell", "#dc2626", "SELL - WEAK OUTLOOK"
        else:
            return "💀 STRONG SELL", "strong-sell", "#991b1b", "STRONG SELL - AVOID COMPLETELY"

    def create_advanced_chart(self, df, symbol):
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{symbol} - ADVANCED ANALYSIS', 
                'Volume & Money Flow',
                'RSI Multi-Timeframe', 
                'MACD & Trend Strength'
            ),
            row_heights=[0.4, 0.15, 0.2, 0.25]
        )
        
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#00C805', decreasing_line_color='#FF0000'
        ), row=1, col=1)
        
        for period, color in [(8, 'orange'), (21, 'green'), (55, 'blue'), (144, 'purple')]:
            if f'EMA_{period}' in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{period}'], 
                                       name=f'EMA {period}', line=dict(color=color, width=2)), row=1, col=1)
        
        if 'BB_Upper_2' in df and 'BB_Lower_2' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper_2'], name='BB Upper', 
                                   line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower_2'], name='BB Lower', 
                                   line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
        
        colors = ['#00C805' if row['Close'] >= row['Open'] else '#FF0000' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                           marker_color=colors, opacity=0.7), row=2, col=1)
        
        if 'Volume_SMA_20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA_20'], name='Vol SMA 20', 
                                   line=dict(color='yellow', width=2)), row=2, col=1)
        
        for period, color in [(14, 'blue'), (21, 'green')]:
            if f'RSI_{period}' in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[f'RSI_{period}'], name=f'RSI {period}', 
                                       line=dict(color=color, width=2)), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        if all(col in df for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                                   line=dict(color='blue', width=2)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                                   line=dict(color='red', width=2)), row=4, col=1)
            
            colors_hist = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', 
                               marker_color=colors_hist), row=4, col=1)
            
            fig.add_hline(y=0, line_color="black", row=4, col=1)
        
        fig.update_layout(
            title=f'ADVANCED TRADING ANALYSIS - {symbol}',
            height=1000,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

def main():
    app = UltraStockAnalyzerPro()
    
    st.markdown('<h1 class="main-header">🚀 MY PRIVATE ULTRA STOCK ANALYZER PRO <span class="private-badge">PRIVATE</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #6b7280;">🔒 Personal Use Only - 40+ Enhanced Indicators</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ANALYSIS PARAMETERS")
        
        selected_stock = st.selectbox("Select Stock:", list(app.all_symbols.keys()))
        symbol = app.all_symbols[selected_stock]
        
        custom_symbol = st.text_input("Or Enter Custom Symbol:", placeholder="e.g., TATAMOTORS.NS")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        st.subheader("💰 TRADING SETTINGS")
        col_a, col_b = st.columns(2)
        
        with col_a:
            stop_loss_percent = st.slider("Stop Loss %", 1.0, 20.0, 8.0, 0.5)
            risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, 1.5, 0.1)
        
        with col_b:
            target_percent = st.slider("Target %", 1.0, 50.0, 15.0, 1.0)
            position_size = st.slider("Position Size (%)", 1, 100, 10)
        
        st.subheader("🧮 POSITION CALCULATOR")
        with st.expander("Open Calculator"):
            capital = st.number_input("Total Capital (₹)", min_value=1000, value=100000, step=1000)
            risk_per_trade = st.number_input("Risk per Trade (%)", min_value=0.1, value=2.0, step=0.1)
            
            if st.button("Calculate Position"):
                data = app.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    stop_loss = current_price * (1 - stop_loss_percent/100)
                    
                    shares, position_value = app.calculate_position_size(
                        capital, risk_per_trade, current_price, stop_loss
                    )
                    
                    st.markdown(f'''
                    <div class="calculator-box">
                        <h3>💰 POSITION CALCULATION</h3>
                        <p><strong>Current Price:</strong> ₹{current_price:.2f}</p>
                        <p><strong>Stop Loss:</strong> ₹{stop_loss:.2f}</p>
                        <p><strong>Shares to Buy:</strong> {shares}</p>
                        <p><strong>Position Value:</strong> ₹{position_value:,.2f}</p>
                        <p><strong>Risk Amount:</strong> ₹{capital * (risk_per_trade/100):,.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
        
        if st.button("🚀 RUN ADVANCED ANALYSIS", type="primary", use_container_width=True):
            with st.spinner("🔄 Running enhanced analysis..."):
                data = app.get_stock_data(symbol, "1y")
                
                if data is not None and not data.empty:
                    df = app.calculate_advanced_indicators(data)
                    
                    if df is not None:
                        score, reasons, signals, bullish_count, total_signals = app.calculate_ai_score(df)
                        signal, signal_class, color, advice = app.get_trading_signal(score)
                        current_price = df['Close'].iloc[-1]
                        
                        stop_loss_price = current_price * (1 - stop_loss_percent/100)
                        target_price = current_price * (1 + target_percent/100)
                        actual_risk_reward = (target_price - current_price) / (current_price - stop_loss_price)
                        
                        st.markdown(f'<div class="super-card {signal_class}">', unsafe_allow_html=True)
                        st.subheader(f"🎯 {signal}")
                        st.write(f"**AI Confidence Score:** {score}/100")
                        st.write(f"**Current Price:** ₹{current_price:.2f}")
                        st.write(f"**Bullish Signals:** {bullish_count}/{total_signals}")
                        st.write(f"**Analysis:** {advice}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div class="trade-box">
                            <h3>💰 TRADING RECOMMENDATION</h3>
                            <p><strong>Entry Price:</strong> ₹{current_price:.2f}</p>
                            <p><strong>Stop Loss:</strong> ₹{stop_loss_price:.2f} ({stop_loss_percent}%)</p>
                            <p><strong>Target Price:</strong> ₹{target_price:.2f} (+{target_percent}%)</p>
                            <p><strong>Risk/Reward:</strong> 1:{actual_risk_reward:.1f}</p>
                            <p><strong>Position Size:</strong> {position_size}% of capital</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.subheader("📊 ENHANCED INDICATOR SIGNALS")
                        cols = st.columns(4)
                        
                        indicators = []
                        if 'RSI_14' in df:
                            rsi_val = df['RSI_14'].iloc[-1]
                            rsi_status = 'bullish' if 40 <= rsi_val <= 65 else 'bearish'
                            indicators.append(('RSI 14', f"{rsi_val:.1f}", rsi_status))
                        
                        if all(col in df for col in ['MACD', 'MACD_Signal']):
                            macd_status = 'bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'bearish'
                            indicators.append(('MACD', 'BULLISH' if macd_status == 'bullish' else 'BEARISH', macd_status))
                        
                        if all(col in df for col in ['Volume', 'Volume_SMA_20']):
                            vol_ratio = df['Volume'].iloc[-1] / df['Volume_SMA_20'].iloc[-1]
                            vol_status = 'bullish' if vol_ratio > 1.5 else 'neutral'
                            indicators.append(('Volume', f"{vol_ratio:.1f}x", vol_status))
                        
                        if 'ADX' in df:
                            adx_val = df['ADX'].iloc[-1]
                            trend_status = 'bullish' if adx_val > 25 else 'neutral'
                            indicators.append(('Trend Power', f"{adx_val:.1f}", trend_status))
                        
                        if 'BB_Position' in df:
                            bb_pos = df['BB_Position'].iloc[-1]
                            bb_status = 'bullish' if bb_pos < 0.3 else 'neutral'
                            indicators.append(('BB Position', f"{bb_pos:.2f}", bb_status))
                        
                        if 'MFI' in df:
                            mfi_val = df['MFI'].iloc[-1]
                            mfi_status = 'bullish' if 20 <= mfi_val <= 80 else 'neutral'
                            indicators.append(('MFI', f"{mfi_val:.1f}", mfi_status))
                        
                        if 'TEMA' in df:
                            tema_status = 'bullish' if current_price > df['TEMA'].iloc[-1] else 'bearish'
                            indicators.append(('TEMA', 'BULL' if tema_status == 'bullish' else 'BEAR', tema_status))
                        
                        if 'OBV_EMA' in df:
                            obv_status = 'bullish' if df['OBV'].iloc[-1] > df['OBV_EMA'].iloc[-1] else 'neutral'
                            indicators.append(('OBV Trend', 'RISING' if obv_status == 'bullish' else 'FALLING', obv_status))
                        
                        for idx, (name, value, status) in enumerate(indicators[:8]):
                            with cols[idx % 4]:
                                st.markdown(f'''
                                <div class="indicator-box {status}">
                                    <h4>{name}</h4>
                                    <h3>{value}</h3>
                                </div>
                                ''', unsafe_allow_html=True)
            
            with col2:
                if 'df' in locals() and df is not None:
                    st.plotly_chart(app.create_advanced_chart(df, selected_stock), use_container_width=True)
                    
                    st.subheader("🔍 ENHANCED ANALYSIS REPORT")
                    for i, reason in enumerate(reasons[:15], 1):
                        st.write(f"{i}. {reason}")
                    
                    st.subheader("🛡️ ADVANCED RISK MANAGEMENT")
                    risk_cols = st.columns(6)
                    
                    with risk_cols[0]:
                        st.metric("Stop Loss", f"₹{stop_loss_price:.1f}")
                    
                    with risk_cols[1]:
                        st.metric("Target Price", f"₹{target_price:.1f}")
                    
                    with risk_cols[2]:
                        st.metric("Risk/Reward", f"1:{actual_risk_reward:.1f}")
                    
                    with risk_cols[3]:
                        if 'ATR' in df:
                            st.metric("ATR", f"₹{df['ATR'].iloc[-1]:.2f}")
                    
                    with risk_cols[4]:
                        if 'Support_1' in df:
                            st.metric("Support", f"₹{df['Support_1'].iloc[-1]:.1f}")
                    
                    with risk_cols[5]:
                        if 'Resistance_1' in df:
                            st.metric("Resistance", f"₹{df['Resistance_1'].iloc[-1]:.1f}")

    st.sidebar.header("⚡ QUICK ACTIONS")
    if st.sidebar.button("🔍 SCAN ALL STOCKS"):
        with st.spinner("Scanning all stocks..."):
            results = []
            progress_bar = st.sidebar.progress(0)
            
            for i, (stock_name, stock_symbol) in enumerate(app.all_symbols.items()):
                try:
                    data = app.get_stock_data(stock_symbol)
                    if data is not None and not data.empty:
                        df = app.calculate_advanced_indicators(data)
                        if df is not None:
                            score, _, _, bullish_count, _ = app.calculate_ai_score(df)
                            if score >= 80:
                                current_price = df['Close'].iloc[-1]
                                results.append({
                                    'symbol': stock_name,
                                    'price': current_price,
                                    'score': score,
                                    'bullish': bullish_count
                                })
                    progress_bar.progress((i + 1) / len(app.all_symbols))
                except:
                    continue
            
            if results:
                st.subheader("💎 TOP RECOMMENDATIONS")
                for stock in sorted(results, key=lambda x: x['score'], reverse=True)[:5]:
                    signal_class = "ultra-buy" if stock['score'] >= 90 else "strong-buy"
                    st.markdown(f'''
                    <div class="super-card {signal_class}">
                        <h3>🚀 {stock['symbol']}</h3>
                        <p><strong>AI Score:</strong> {stock['score']}/100</p>
                        <p><strong>Current Price:</strong> ₹{stock['price']:.2f}</p>
                        <p><strong>Bullish Signals:</strong> {stock['bullish']}</p>
                    </div>
                    ''', unsafe_allow_html=True)

    st.sidebar.header("🔐 SECURITY")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    st.sidebar.info("""
    **Private Access Only**
    - Personal use
    - Not for public sharing  
    - Secure analysis
    - Enhanced accuracy
    """)

if __name__ == "__main__":
    main()
