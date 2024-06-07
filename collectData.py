import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import tensorflow as tf
import subprocess

# Function to fetch and save stock data
def fetch_and_save_stock_data(tickers, start_date, end_date, interval='1m'):
    all_data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        # Create directory for the ticker if it doesn't exist
        ticker_dir = os.path.join('stock_data', ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Save the data to a CSV file
        data_filename = os.path.join(ticker_dir, f"{ticker}_data_{start_date}_to_{end_date}_1m.csv")
        stock_data.to_csv(data_filename)
        print(f"Stock data saved to {data_filename}")
        
        all_data[ticker] = stock_data
    return all_data

# Simulate 100 trades
def simulate_trades(stock_data, num_trades=100, shares_per_trade=10):
    trades = []
    for ticker, data in stock_data.items():
        data_len = len(data)
        if data_len < 2:
            print(f"Not enough data to simulate trades for {ticker}.")
            continue

        for _ in range(num_trades):
            buy_index = random.randint(0, data_len - 2)
            buy_date = data.index[buy_index]
            buy_price = data['Close'].iloc[buy_index]
            sell_index = random.randint(buy_index + 1, data_len - 1)
            sell_date = data.index[sell_index]
            sell_price = data['Close'].iloc[sell_index]
            profit_loss = (sell_price - buy_price) * shares_per_trade
            percentage_change = (sell_price - buy_price) / buy_price * 100
            score = percentage_change
            
            # Extract the past 30 minutes of data
            start_index = max(0, buy_index - 30)
            past_30min_data = data.iloc[start_index:buy_index]

            trades.append({
                'Ticker': ticker,
                'Buy Date': buy_date,
                'Sell Date': sell_date,
                'Buy Index': buy_index,
                'Sell Index': sell_index,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Shares': shares_per_trade,
                'Profit/Loss': profit_loss,
                'Percentage Change': percentage_change,
                'Score': score,
                'Past 30min Data': past_30min_data
            })
    
    # Normalize the scores globally
    scores = [trade['Score'] for trade in trades]
    min_score = min(scores)
    max_score = max(scores)
    for trade in trades:
        trade['Normalized Score'] = (trade['Score'] - min_score) / (max_score - min_score) if max_score != min_score else 0
    
    return trades

# Function to write trades to TensorBoard
def write_trades_to_tensorboard(trades, log_dir):
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        for i, trade in enumerate(trades):
            tf.summary.scalar('Buy Price', trade['Buy Price'], step=i)
            tf.summary.scalar('Sell Price', trade['Sell Price'], step=i)
            tf.summary.scalar('Profit/Loss', trade['Profit/Loss'], step=i)
            tf.summary.scalar('Percentage Change', trade['Percentage Change'], step=i)
            tf.summary.scalar('Normalized Score', trade['Normalized Score'], step=i)
            for j, (index, row) in enumerate(trade['Past 30min Data'].iterrows()):
                tf.summary.scalar(f'Past 30min Close {j}', row['Close'], step=i)
    writer.close()

# Function to start TensorBoard
def start_tensorboard(log_dir):
    print(f"Starting TensorBoard on logdir {log_dir}")
    subprocess.Popen(['tensorboard', '--logdir', log_dir])

# Define the tickers and date range
tickers = ['AAPL', 'AMZN']
end_date = datetime.today().date()
start_date = end_date - timedelta(days=7)  # Last 7 days (1 min data limit)

def main():
    print("Select an operation:")
    print("1. Fetch and save stock data")
    print("2. Simulate trades")
    print("3. Write trades to TensorBoard")
    print("4. Write trades to TensorBoard and start TensorBoard")
    choice = input("Enter choice (1-4): ")

    if choice == '1':
        stock_data = fetch_and_save_stock_data(tickers, start_date, end_date, interval='1m')
        print("Stock data fetched and saved.")
    elif choice == '2':
        stock_data = fetch_and_save_stock_data(tickers, start_date, end_date, interval='1m')
        trades = simulate_trades(stock_data)
        print("Trades simulated.")
    elif choice == '3':
        stock_data = fetch_and_save_stock_data(tickers, start_date, end_date, interval='1m')
        trades = simulate_trades(stock_data)
        log_dir = "logs/trades"
        write_trades_to_tensorboard(trades, log_dir)
        print(f"Trades written to TensorBoard logs in {log_dir}")
    elif choice == '4':
        stock_data = fetch_and_save_stock_data(tickers, start_date, end_date, interval='1m')
        trades = simulate_trades(stock_data)
        log_dir = "logs/trades"
        write_trades_to_tensorboard(trades, log_dir)
        start_tensorboard(log_dir)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

