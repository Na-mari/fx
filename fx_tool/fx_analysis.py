# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output, callback_context
import sqlite3
import traceback
from datetime import datetime
from pytz import timezone

# UTCの日時を日本時間に変換
def convert_to_japan_time(utc_time):
    utc_zone = timezone('UTC')
    jst_zone = timezone('Asia/Tokyo')
    utc_dt = utc_zone.localize(utc_time)
    jst_dt = utc_dt.astimezone(jst_zone)
    return jst_dt

def create_db():
    con = sqlite3.connect('trading_history.db')
    cursor = con.cursor()

    # 売買履歴テーブルの作成
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        action TEXT,
                        price REAL,
                        profit REAL)''')

    con.commit()
    con.close()

def add_trade(action, price, profit):
    con = sqlite3.connect('trading_history.db')
    cursor = con.cursor()

    # 現在の日時を日本時間で取得
    jst_zone = timezone('Asia/Tokyo')
    jst_now = datetime.now(jst_zone).strftime('%Y-%m-%d %H:%M:%S')
    
    # 直前の取引価格を取得
    cursor.execute("SELECT price, action FROM trades ORDER BY id DESC LIMIT 1")
    last_trade = cursor.fetchone()

    # 利益計算: 売りの場合 (利益 = 売値 - 買値)、買いの場合は負の値
    profit = 0
    if last_trade:
        last_price, last_action = last_trade
        if action == 'Sell' and last_action == 'Buy':
            profit = price - last_price
        elif action == 'Buy' and last_action == 'Sell':
            profit = last_price - price

    # 取引情報の挿入
    cursor.execute("INSERT INTO trades (date, action, price, profit) VALUES (?, ?, ?, ?)",
                   (jst_now, action, price, profit))
    
    con.commit()
    con.close()

def calculate_total_profit():
    con = sqlite3.connect('trading_history.db')
    cursor = con.cursor()

    # 総利益の計算
    cursor.execute("SELECT SUM(profit) FROM trades")
    total_profit = cursor.fetchone()[0]
    con.close()
    
    return total_profit if total_profit else 0

# ===== データ取得 =====
def fetch_fx_data(pair='USDJPY=X', period='1y', interval='1d'):
    """
    Yahoo Financeから為替データを取得
    :param pair: 通貨ペア（例：USDJPY=X）
    :param period: データ期間（例：1y, 5d）
    :param interval: 時間間隔（例：1d, 1h）
    """
    data = yf.download(pair, period=period, interval=interval)
    return data

# ===== テクニカル分析 =====
def calculate_sma(data, window):
    """移動平均線を計算"""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """RSIを計算"""
    close_prices = data['Close']  # ここで Series 型を確保
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===== バックテストとシグナル生成 =====
def generate_signals(data):
    """移動平均クロスオーバー戦略でシグナル生成"""
    close_prices = data['Close']  # Series 型を使用
    data['ShortSMA'] = calculate_sma(data, 10)
    data['LongSMA'] = calculate_sma(data, 50)
    data['Signal'] = 0
    data.loc[data['ShortSMA'] > data['LongSMA'], 'Signal'] = 1
    data.loc[data['ShortSMA'] <= data['LongSMA'], 'Signal'] = -1

    # シグナルの位置を記録
    data['BuySignal'] = close_prices.where(data['Signal'] == 1)
    data['SellSignal'] = close_prices.where(data['Signal'] == -1)
    return data


def backtest_strategy(data):
    """バックテスト実行"""
    data['DailyReturn'] = data['Close'].pct_change()
    data['StrategyReturn'] = data['Signal'].shift(1) * data['DailyReturn']
    total_return = data['StrategyReturn'].cumsum().iloc[-1]
    return total_return

# ===== 可視化 =====
def plot_data_with_signals(data):
    """データをプロットし、売買シグナルを表示"""
    plt.figure(figsize=(12, 8))

    # 為替レートと移動平均線
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['ShortSMA'], label='Short SMA (10)', color='orange')
    plt.plot(data['LongSMA'], label='Long SMA (50)', color='green')

    # 売買シグナル
    plt.scatter(data.index, data['BuySignal'], label='Buy Signal', color='green', marker='^', alpha=1)
    plt.scatter(data.index, data['SellSignal'], label='Sell Signal', color='red', marker='v', alpha=1)

    plt.title('USD/JPY Exchange Rate with Buy/Sell Signals')
    plt.legend()
    plt.show()

def plot_rsi_with_alerts(data):
    """RSIチャートに通知を追加"""
    plt.figure(figsize=(12, 4))
    plt.plot(data['RSI14'], label='RSI 14', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='blue', linestyle='--', label='Oversold')

    # RSIの通知位置をプロット
    overbought = data[data['RSI14'] > 70]
    oversold = data[data['RSI14'] < 30]
    plt.scatter(overbought.index, overbought['RSI14'], color='red', label='RSI > 70', marker='o')
    plt.scatter(oversold.index, oversold['RSI14'], color='blue', label='RSI < 30', marker='o')

    plt.title('RSI with Alerts')
    plt.legend()
    plt.show()

def get_latest_trades(limit=3):
    con = sqlite3.connect('trading_history.db')
    cursor = con.cursor()

    # 最新の3件の取引を取得
    cursor.execute("SELECT * FROM trades ORDER BY date DESC LIMIT ?", (limit,))
    trades = cursor.fetchall()
    con.close()

    # 必要であれば出力フォーマットを統一
    formatted_trades = [
        (trade[0], datetime.strptime(trade[1], '%Y-%m-%d %H:%M:%S').strftime('%Y/%m/%d %H:%M:%S'), trade[2], trade[3], trade[4])
        for trade in trades
    ]

    return formatted_trades  

def run_dashboard(data, total_return):
    app = Dash(__name__)

    # 初期取引履歴の取得
    latest_trades = get_latest_trades()
    trades_display = [
        f"Date: {trade[1]}, Action: {trade[2]}, Price: {trade[3]:.2f}, Profit: {trade[4]:.2f}"
        for trade in latest_trades
    ]

    # ダッシュボードレイアウト
    app.layout = html.Div([
        html.H1("USD/JPY Analysis Dashboard", style={'fontSize': 24}),
        
        html.Div([
            html.H3(f"移動平均クロスオーバー戦略による総利益: {total_return:.2f}%", style={'textAlign': 'center', 'fontSize': 18}),
        ], style={'textAlign': 'center', 'marginBottom': 20}),
    
        # 総利益表示
        html.Div(id='profit-display', style={'textAlign': 'center', 'fontSize': 18},
                children=f'純利益: {calculate_total_profit():.2f}'),

        # 表示内容（取引履歴や利益）
        html.Div(id='trade-history', children=[
            html.H4("直近の取引履歴"),
            html.Ul([html.Li(trade) for trade in trades_display])
        ]),

        # 売買ボタン
        html.Div([
            html.Button('買', id='buy-button', n_clicks=0),
            html.Button('売', id='sell-button', n_clicks=0)
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # グラフの表示 (価格チャート)
        html.Div([
            dcc.Graph(
                id='price-graph',
                figure={
                    'data': [
                        {'x': data.index.tolist(), 'y': data['Close'].squeeze().tolist(), 'type': 'line', 'name': 'Close Price'},
                        {'x': data.index.tolist(), 'y': data['ShortSMA'].squeeze().tolist(), 'type': 'line', 'name': 'Short SMA (10)'},
                        {'x': data.index.tolist(), 'y': data['LongSMA'].squeeze().tolist(), 'type': 'line', 'name': 'Long SMA (50)'},
                        {'x': data.index.tolist(), 'y': data['BuySignal'].squeeze().tolist(), 'mode': 'markers', 'marker': {'color': 'green', 'symbol': 'triangle-up'}, 'name': 'Buy Signal'},
                        {'x': data.index.tolist(), 'y': data['SellSignal'].squeeze().tolist(), 'mode': 'markers', 'marker': {'color': 'red', 'symbol': 'triangle-down'}, 'name': 'Sell Signal'},
                    ],
                    'layout': {'title': 'USD/JPY with Buy/Sell Signals'}
                }
            )
        ]),

        # グラフの表示 (RSIチャート)
        html.Div([
            dcc.Graph(
                id='rsi-graph',
                figure={
                    'data': [
                        {'x': data.index.tolist(), 'y': data['RSI14'].squeeze().tolist(), 'type': 'line', 'name': 'RSI (14)'},
                        {'x': data.index.tolist(), 'y': [70] * len(data), 'type': 'line', 'name': 'Overbought (70)', 'line': {'dash': 'dash', 'color': 'red'}},
                        {'x': data.index.tolist(), 'y': [30] * len(data), 'type': 'line', 'name': 'Oversold (30)', 'line': {'dash': 'dash', 'color': 'blue'}},
                    ],
                    'layout': {'title': 'RSI with Alerts'}
                }
            )
        ])
    ])

    # コールバック: 売買ボタンがクリックされたときに取引を記録し、利益を更新
    @app.callback(
        Output('trade-history', 'children'),
        [Input('buy-button', 'n_clicks'), Input('sell-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_profit(buy_clicks, sell_clicks):
        try:
            # コンテキストからトリガーされたボタンを確認
            ctx = callback_context
            action = None
            if ctx.triggered:
                if ctx.triggered[0]['prop_id'] == 'buy-button.n_clicks':
                    action = 'Buy'
                elif ctx.triggered[0]['prop_id'] == 'sell-button.n_clicks':
                    action = 'Sell'

            if action:
                # 現在の価格を取得（Seriesから値を抽出して数値化）
                current_price = float(data['Close'].iloc[-1].item())

                # 利益を計算
                profit = float(current_price * (1 if action == 'Sell' else -1))

                # データベースに取引記録を追加
                add_trade(action, current_price, profit)

            # 最新の3件の取引を取得
            latest_trades = get_latest_trades()

            # 取引履歴を表示
            trades_display = []
            for trade in latest_trades:
                trades_display.append(f"Date: {trade[1]}, Action: {trade[2]}, Price: {trade[3]:.2f}, Profit: {trade[4]:.2f}")

            # 総利益を計算
            total_profit = calculate_total_profit()

            # HTML内容を返す
            return (
                html.Div([  # 最新取引履歴
                    html.H4("直近の取引履歴"),
                    html.Ul([html.Li(trade) for trade in trades_display])
                ])
            ) 

        except Exception as e:
            print("Error in update_profit:")
            print(traceback.format_exc())
            return "An error occurred during the update.", ""

    # ダッシュボードの起動
    app.run_server(debug=True)

    # suppress_callback_exceptions を設定 (必要なら)
    app.config.suppress_callback_exceptions = True
    
# ===== メイン処理 =====
if __name__ == "__main__":
    # データ取得
    data = fetch_fx_data()

    # テクニカル指標とシグナル生成
    data['RSI14'] = calculate_rsi(data, 14)
    data = generate_signals(data)

    # バックテスト実行
    total_return = backtest_strategy(data)

    # # チャート表示
    # plot_data_with_signals(data)
    # plot_rsi_with_alerts(data)

    # ダッシュボード起動
    run_dashboard(data, total_return)
