import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedTradingSystem:
    def __init__(self, symbol='AAPL', period='1y', use_hyperparameter_tuning=True, trading_type='swing'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.model = None
        self.features = None
        self.target = None
        self.best_params_ = None
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.trading_type = trading_type
        self.signals = None
        self.price_targets = None
        self.portfolio_value = []
        
    def fetch_data(self):
        """डेटा डाउनलोड करो"""
        print("📊 डेटा डाउनलोड हो रहा है...")
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if self.data.empty:
                raise ValueError("No data received from Yahoo Finance")
            print(f"✅ {len(self.data)} डेटा पॉइंट्स लोड हुए")
            return self.data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def create_features(self):
        """70+ features बनाओ"""
        print("🛠️ Features बन रहे हैं...")
        df = self.data.copy()
        
        try:
            # Price-based features
            df['returns_1d'] = df['Close'].pct_change()
            df['returns_5d'] = df['Close'].pct_change(5)
            df['price_sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['price_sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['price_sma_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            df['sma_20_ratio'] = df['Close'] / df['price_sma_20']
            df['sma_50_ratio'] = df['Close'] / df['price_sma_50']
            
            # Momentum features
            for period in [5, 10, 14]:
                df[f'roc_{period}'] = ta.momentum.roc(df['Close'], window=period)
                df[f'rsi_{period}'] = ta.momentum.rsi(df['Close'], window=period)
            
            # Stochastic
            df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

            # Trend features
            macd = ta.trend.macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = ta.trend.macd_signal(df['Close'])
            df['macd_histogram'] = ta.trend.macd_diff(df['Close'])
            df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            df['dpo'] = ta.trend.dpo(df['Close'])

            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_above_cloud'] = (df['Close'] > df['ichimoku_a']) & (df['Close'] > df['ichimoku_b'])
            
            # Volatility features
            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            df['atr_pct'] = df['atr'] / df['Close']
            
            bb = ta.volatility.BollingerBands(df['Close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            df['volume_sma'] = ta.trend.sma_indicator(df['Volume'], window=20)
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['mfi'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            
            # Support/Resistance
            for window in [20, 50]:
                df[f'resistance_{window}'] = df['High'].rolling(window).max()
                df[f'support_{window}'] = df['Low'].rolling(window).min()
                df[f'distance_to_resistance_{window}'] = (df[f'resistance_{window}'] - df['Close']) / df['Close']
                df[f'distance_to_support_{window}'] = (df['Close'] - df[f'support_{window}']) / df['Close']
            
            # Price patterns
            df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
            df['open_close_range'] = (df['Close'] - df['Open']) / df['Open']
            df['body_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().dropna()
            
            self.features = [col for col in df.columns if col not in 
                           ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'future_close', 'future_return']]
            
            print(f"✅ {len(self.features)} features बन गए हैं")
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return None
    
    def create_target(self, df):
        """Target variable बनाओ - 2% profit in specified period"""
        if self.trading_type == 'intraday':
            lookahead = 1
            profit_threshold = 0.01  # 1% for intraday
        else:
            lookahead = 5
            profit_threshold = 0.02  # 2% for swing
            
        df['future_close'] = df['Close'].shift(-lookahead)
        df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
        df['target'] = (df['future_return'] > profit_threshold).astype(int)
        df = df.dropna()
        
        print(f"✅ Target created: {lookahead}-day {profit_threshold:.1%} profit threshold")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        return df

    def calculate_price_targets(self, df, predictions, confidence_scores):
        """Smart price targets calculate करो"""
        print("🎯 Price Targets Calculate हो रहे हैं...")
        
        targets = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= len(predictions): 
                continue
                
            current_price = row['Close']
            prediction = predictions[i]
            confidence = confidence_scores[i]
            atr = row['atr']
            
            # Different parameters for trading styles
            if self.trading_type == 'intraday':
                target_multiplier = 1.0 + (confidence - 0.5) * 0.5  # 0.75x to 1.25x
                stop_multiplier = 0.6
                max_risk_reward = 3.0
            else:
                target_multiplier = 1.5 + (confidence - 0.5) * 1.0  # 1.0x to 2.0x
                stop_multiplier = 0.8
                max_risk_reward = 4.0
            
            if prediction == 1:  # BUY signal
                # Base target from ATR
                base_target = current_price + (atr * target_multiplier)
                
                # Consider resistance levels
                resistance_20 = row.get('resistance_20', current_price * 1.1)
                resistance_50 = row.get('resistance_50', current_price * 1.15)
                
                # Smart target selection
                target_price = min(base_target, resistance_20)
                if confidence > 0.7 and base_target < resistance_50:
                    target_price = min(base_target * 1.1, resistance_50)
                
                # Stop loss below support or ATR-based
                support_20 = row.get('support_20', current_price * 0.95)
                stop_loss = max(current_price - (atr * stop_multiplier), support_20 * 0.99)
                
            else:  # SELL signal
                # Base target from ATR
                base_target = current_price - (atr * target_multiplier)
                
                # Consider support levels
                support_20 = row.get('support_20', current_price * 0.9)
                support_50 = row.get('support_50', current_price * 0.85)
                
                # Smart target selection
                target_price = max(base_target, support_20)
                if confidence > 0.7 and base_target > support_50:
                    target_price = max(base_target * 0.9, support_50)
                
                # Stop loss above resistance or ATR-based
                resistance_20 = row.get('resistance_20', current_price * 1.05)
                stop_loss = min(current_price + (atr * stop_multiplier), resistance_20 * 1.01)
            
            # Calculate metrics
            potential_profit = abs(target_price - current_price)
            risk = abs(current_price - stop_loss)
            risk_reward = min(potential_profit / risk, max_risk_reward) if risk > 0 else 1
            
            # Adjust for confidence
            confidence_boost = 1.0 + (confidence - 0.5) * 0.3
            target_price = target_price * confidence_boost if prediction == 1 else target_price / confidence_boost

            targets.append({
                'date': idx,
                'current_price': current_price,
                'signal': 'BUY' if prediction == 1 else 'SELL',
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'confidence': confidence,
                'potential_profit_pct': round((potential_profit / current_price) * 100, 2),
                'risk_reward_ratio': round(risk_reward, 2),
                'atr_pct': round((atr / current_price) * 100, 2)
            })
        
        self.price_targets = pd.DataFrame(targets)
        return self.price_targets

    def train_model(self, df):
        """Advanced model training"""
        if df is None or len(df) < 100:
            print("❌ Insufficient data for training")
            return None, None, None, None
            
        X = df[self.features]
        y = df['target']
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        try:
            if self.use_hyperparameter_tuning:
                print("🤖 OPTIMIZED ML Model training शुरू...")
                tscv = TimeSeriesSplit(n_splits=3)
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10]
                }
                
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid, 
                    cv=tscv, 
                    n_jobs=-1, 
                    scoring='accuracy',
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                print(f"🏆 Best Parameters: {self.best_params_}")
                print(f"🎯 Best CV Score: {grid_search.best_score_:.2%}")
                
            else:
                print("🤖 BASIC ML Model training शुरू...")
                self.model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(X_train, y_train)

            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Training Complete")
            print(f"📊 Test Accuracy: {accuracy:.2%}")
            
            # Additional metrics
            buy_accuracy = accuracy_score(y_test[y_test==1], y_pred[y_test==1]) if len(y_test[y_test==1]) > 0 else 0
            sell_accuracy = accuracy_score(y_test[y_test==0], y_pred[y_test==0]) if len(y_test[y_test==0]) > 0 else 0
            
            print(f"📈 Buy Accuracy: {buy_accuracy:.2%} | Sell Accuracy: {sell_accuracy:.2%}")
            
            return X_test, y_test, y_pred, y_pred_proba
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return None, None, None, None

    def generate_trading_signals(self, test_df, y_pred, y_pred_proba):
        """Generate trading signals with position management"""
        print("💰 Trading Signals Generate हो रहे हैं...")
        
        signals = []
        position = None
        entry_price = 0
        entry_date = None
        capital = 10000
        position_size = 0
        
        for i, (idx, row) in enumerate(test_df.iterrows()):
            current_price = row['Close']
            prediction = y_pred[i] if i < len(y_pred) else 0
            confidence = y_pred_proba[i] if i < len(y_pred_proba) else 0.5
            
            # Get current signal
            current_signal = 'BUY' if prediction == 1 else 'SELL'
            
            # Signal logic with confidence threshold
            if position is None:  # No position
                if current_signal == 'BUY' and confidence > 0.65:
                    action = 'BUY'
                    position = 'LONG'
                    entry_price = current_price
                    entry_date = idx
                    position_size = capital / current_price
                    capital = 0
                elif current_signal == 'SELL' and confidence > 0.65:
                    action = 'SELL'
                    position = 'SHORT'
                    entry_price = current_price
                    entry_date = idx
                    position_size = capital / current_price
                    capital = 0
                else:
                    action = 'HOLD'
                    
            elif position == 'LONG':  # In long position
                # Check exit conditions
                target_price = row.get('target_price', entry_price * 1.02)
                stop_loss = row.get('stop_loss', entry_price * 0.98)
                
                if current_price >= target_price or current_price <= stop_loss:
                    action = 'SELL'
                    capital = position_size * current_price
                    position = None
                    position_size = 0
                else:
                    action = 'HOLD'
                    
            elif position == 'SHORT':  # In short position
                # Check exit conditions
                target_price = row.get('target_price', entry_price * 0.98)
                stop_loss = row.get('stop_loss', entry_price * 1.02)
                
                if current_price <= target_price or current_price >= stop_loss:
                    action = 'BUY'
                    capital = position_size * (2 * entry_price - current_price)  # P&L for short
                    position = None
                    position_size = 0
                else:
                    action = 'HOLD'
            
            # Calculate current portfolio value
            if position == 'LONG':
                current_value = position_size * current_price
            elif position == 'SHORT':
                current_value = position_size * (2 * entry_price - current_price)
            else:
                current_value = capital
                
            self.portfolio_value.append(current_value)
            
            signals.append({
                'date': idx,
                'price': current_price,
                'signal': action,
                'position': position,
                'entry_price': entry_price,
                'position_size': position_size,
                'portfolio_value': current_value,
                'confidence': confidence,
                'target_price': row.get('target_price', 0),
                'stop_loss': row.get('stop_loss', 0)
            })
        
        return pd.DataFrame(signals)

    def backtest(self, test_df, signals_df, initial_capital=10000):
        """Comprehensive backtesting"""
        print("\n🔍 Backtesting शुरू...")
        
        if signals_df.empty:
            print("❌ No signals to backtest")
            return None
            
        final_capital = signals_df['portfolio_value'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate additional metrics
        trades = signals_df[signals_df['signal'].isin(['BUY', 'SELL'])]
        winning_trades = len(trades[trades['portfolio_value'] > trades['portfolio_value'].shift(1)])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe Ratio
        returns = signals_df['portfolio_value'].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        peak = signals_df['portfolio_value'].cummax()
        drawdown = (signals_df['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        print("📊 BACKTESTING RESULTS:")
        print(f"✅ Initial Capital: ${initial_capital:,.2f}")
        print(f"✅ Final Capital: ${final_capital:,.2f}")
        print(f"✅ Total Return: {total_return:.2%}")
        print(f"✅ Win Rate: {win_rate:.2%}")
        print(f"✅ Total Trades: {total_trades}")
        print(f"✅ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"✅ Max Drawdown: {max_drawdown:.2%}")
        
        return {
            'test_data': test_df,
            'signals': signals_df,
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': final_capital,
            'total_trades': total_trades
        }
    
    def market_regime_detection(self, df):
        """Advanced market regime detection"""
        print("\n🌊 Market Regime Detection...")
        
        # Trend detection
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 'uptrend', 'downtrend')
        
        # Volatility regime
        df['volatility_20'] = df['returns_1d'].rolling(20).std()
        vol_threshold = df['volatility_20'].quantile(0.7)
        df['vol_regime'] = np.where(df['volatility_20'] > vol_threshold, 'high_vol', 'low_vol')
        
        # ADX based trend strength
        df['adx_regime'] = np.where(df['adx'] > 25, 'strong_trend', 'weak_trend')
        
        # Combined regime
        df['market_regime'] = df['trend'] + '_' + df['vol_regime'] + '_' + df['adx_regime']
        
        regime_counts = df['market_regime'].value_counts()
        print("📈 Market Regime Distribution:")
        for regime, count in regime_counts.head().items():
            print(f"   {regime}: {count} days ({count/len(df):.1%})")
        
        return df

    def plot_results(self, results):
        """Comprehensive results plotting"""
        if results is None:
            print("❌ No results to plot")
            return
            
        print("\n📈 Trading Charts बन रहे हैं...")
        
        test_data = results['test_data']
        signals = results['signals']
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price and signals
        axes[0].plot(test_data.index, test_data['Close'], label='Price', color='black', linewidth=1, alpha=0.8)
        
        # Buy signals
        buy_signals = signals[signals['signal'] == 'BUY']
        axes[0].scatter(buy_signals['date'], buy_signals['price'], 
                       color='green', marker='^', s=100, label='BUY', zorder=5)
        
        # Sell signals
        sell_signals = signals[signals['signal'] == 'SELL']
        axes[0].scatter(sell_signals['date'], sell_signals['price'], 
                       color='red', marker='v', s=100, label='SELL', zorder=5)
        
        # Price targets and stop losses
        for _, target in self.price_targets.iterrows():
            color = 'green' if target['signal'] == 'BUY' else 'red'
            axes[0].plot([target['date'], target['date']], 
                        [target['current_price'], target['target_price']], 
                        color=color, linestyle='--', alpha=0.7)
            axes[0].scatter(target['date'], target['target_price'], 
                           color=color, marker='*', s=50, zorder=4)
        
        axes[0].set_title(f'{self.symbol} - Trading Signals & Price Targets', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Portfolio value
        axes[1].plot(signals['date'], signals['portfolio_value'], 
                    label='Portfolio Value', color='blue', linewidth=2)
        axes[1].set_title('Portfolio Performance', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Value ($)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Confidence scores
        axes[2].plot(signals['date'], signals['confidence'], 
                    label='Signal Confidence', color='orange', linewidth=1)
        axes[2].axhline(y=0.65, color='red', linestyle='--', label='Confidence Threshold (0.65)')
        axes[2].set_title('Signal Confidence Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Confidence', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def print_trading_recommendation(self, results):
        """Detailed trading recommendation"""
        if results is None:
            print("❌ No results for recommendation")
            return
            
        print("\n" + "="*80)
        print("🎯 CURRENT TRADING RECOMMENDATION")
        print("="*80)
        
        latest_signal = results['signals'].iloc[-1]
        latest_price = results['test_data'].iloc[-1]
        
        print(f"📊 {self.symbol} TRADING UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"   • Current Price: ${latest_signal['price']:.2f}")
        print(f"   • Signal: {latest_signal['signal']}")
        print(f"   • Confidence: {latest_signal['confidence']:.1%}")
        print(f"   • Position: {latest_signal['position']}")
        
        if latest_signal['position']:
            print(f"   • Entry Price: ${latest_signal['entry_price']:.2f}")
            print(f"   • Current P&L: {((latest_signal['price'] - latest_signal['entry_price']) / latest_signal['entry_price'] * 100):.2f}%")
        
        # Find corresponding price target
        current_target = self.price_targets[self.price_targets['date'] == latest_signal['date']]
        if not current_target.empty:
            target = current_target.iloc[0]
            print(f"\n🎯 PRICE TARGETS:")
            print(f"   • Target Price: ${target['target_price']:.2f}")
            print(f"   • Stop Loss: ${target['stop_loss']:.2f}")
            print(f"   • Potential Profit: {target['potential_profit_pct']}%")
            print(f"   • Risk/Reward: {target['risk_reward_ratio']}:1")
            print(f"   • ATR %: {target['atr_pct']}%")
        
        print(f"\n💡 TRADING ADVICE:")
        if latest_signal['signal'] == 'BUY' and latest_signal['confidence'] > 0.65:
            print("   ✅ **STRONG BUY** - Enter LONG position")
            print(f"   🎯 Target: ${target['target_price']:.2f} (+{target['potential_profit_pct']}%)")
            print(f"   🛡️ Stop Loss: ${target['stop_loss']:.2f}")
            print(f"   ⚖️ Risk/Reward: {target['risk_reward_ratio']}:1")
            
        elif latest_signal['signal'] == 'SELL' and latest_signal['confidence'] > 0.65:
            print("   🔻 **STRONG SELL** - Enter SHORT position")
            print(f"   🎯 Target: ${target['target_price']:.2f} (-{target['potential_profit_pct']}%)")
            print(f"   🛡️ Stop Loss: ${target['stop_loss']:.2f}")
            print(f"   ⚖️ Risk/Reward: {target['risk_reward_ratio']}:1")
            
        else:
            print("   ⏸️ **WAIT FOR BETTER ENTRY**")
            print("   📊 Market conditions not optimal for high-confidence trade")
            print("   💡 Wait for confidence > 65% and clear trend direction")
        
        print(f"\n📈 PORTFOLIO PERFORMANCE:")
        print(f"   • Total Return: {results['total_return']:.2%}")
        print(f"   • Win Rate: {results['win_rate']:.2%}")
        print(f"   • Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   • Max Drawdown: {results['max_drawdown']:.2%}")
        
        print("="*80)

    def run_complete_analysis(self):
        """Complete trading analysis"""
        print("🚀 ADVANCED TRADING SYSTEM STARTING...")
        print("=" * 60)
        
        # Fetch and prepare data
        data = self.fetch_data()
        if data is None:
            return None, None
            
        df = self.create_features()
        if df is None:
            return None, None
            
        df = self.create_target(df)
        df = self.market_regime_detection(df)
        
        # Train model and get predictions
        X_test, y_test, y_pred, y_pred_proba = self.train_model(df)
        if X_test is None:
            return None, None
            
        # Prepare test data
        test_df = df.loc[X_test.index].copy()
        
        # Calculate price targets
        price_targets_df = self.calculate_price_targets(test_df, y_pred, y_pred_proba)
        test_df = test_df.merge(price_targets_df, on='date', how='left')
        
        # Generate trading signals
        signals_df = self.generate_trading_signals(test_df, y_pred, y_pred_proba)
        
        # Run backtest
        results = self.backtest(test_df, signals_df)
        
        # Generate outputs
        if results:
            self.plot_results(results)
            self.print_trading_recommendation(results)
            
            # Feature importance
            if self.model is not None:
                feature_importance = pd.DataFrame({
                    'feature': self.features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n🔝 TOP 10 FEATURES:")
                print(feature_importance.head(10).to_string(index=False))
        
        return df, results

# Run the system
if __name__ == "__main__":
    print("Choose your configuration:")
    symbol = input("Enter stock symbol (default AAPL): ").strip() or "AAPL"
    
    print("\nChoose trading style:")
    print("1. SWING Trading (2-5 days holding)")
    print("2. INTRADAY Trading (Same day)")
    trading_choice = input("Enter choice (1 or 2): ").strip()
    trading_type = "swing" if trading_choice == "1" else "intraday"
    
    print("\nChoose model type:")
    print("1. OPTIMIZED (Better accuracy, slower)")
    print("2. BASIC (Faster, good for testing)")
    model_choice = input("Enter choice (1 or 2): ").strip()
    use_tuning = model_choice == "1"
    
    trading_system = AdvancedTradingSystem(
        symbol=symbol,
        period='2y',
        use_hyperparameter_tuning=use_tuning,
        trading_type=trading_type
    )
    
    trading_system.run_complete_analysis()
