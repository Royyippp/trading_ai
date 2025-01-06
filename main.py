import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from technical import TechnicalIndicators
from condition import Condition
from strategy import Strategy
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the CSV data into a DataFrame
df = pd.read_csv('NAS100_M1.csv', parse_dates=['time'])
df.sort_values(by='time', inplace=True)
df.reset_index(drop=True, inplace=True)

# Define the state structure
class State(TypedDict):
    user_query: str
    technical_series: dict
    entry_conditions: list
    exit_conditions: list
    strategy_results: dict
    generated_series: dict

# Initialize the ChatOpenAI model
model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

# Function to generate an example query using the LLM
def generate_example_query():
    prompt = """
    Generate an example trading query for a user who wants to analyze a trading strategy.
    The query should be clear and specific, like:
    "What is the percentage of winning using MACD crossing strategy?" in Chinese Traditional,
    You may choose some indicators like MACD, RSI, SMA, EMA, Bollinger Bands, ATR, ADX, Stochastic Oscillator, Momentum.
    """
    response = model.invoke([HumanMessage(content=prompt)])
    try:
        example_query = response.content.strip()
        return example_query
    except Exception as e:
        print(f"Error generating example query: {e}")
        return "What is the percentage of winning using MACD crossing strategy?"
    
# Step 1: Ask what the user wants
def ask_user_query(state: State):
    """
    Capture the user's query and store it in the state.
    """
    user_query = input("What would you like to analyze? ")
    state['user_query'] = user_query
    return state

# Step 2: Determine what technical series to generate
def determine_technical_series(state: State):
    """
    Use the LLM to determine which technical series are needed based on the query.
    """
    prompt = f"""
        A user asked: "{state['user_query']}"
        Determine which technical series to generate.
        The available technical indicators and their parameters are as follows:
        - MACD: {{"fast": int, "slow": int, "signal": int}}
        - RSI: {{"length": int}}
        - SMA: {{"length": int}}
        - EMA: {{"length": int}}
        - Bollinger_Bands: {{"length": int, "num_std_dev": int}}
        - ATR: {{"length": int}}
        - ADX: {{"length": int}}
        - Stochastic_Oscillator: {{"length": int}}
        - Momentum: {{"length": int}}
        
        Respond with a JSON object listing the series names and any required parameters, like:
        {{
            "series": [
                {{
                    "name": "MACD",
                    "parameters": {{"fast": 12, "slow": 26, "signal": 9}}
                }},
                {{
                    "name": "RSI",
                    "parameters": {{"length": 14}}
                }}
            ]
        }}
        """
    response = model.invoke([HumanMessage(content=prompt)])
    print("Model Response:", response.content)

    # Parse the JSON content from the AIMessage
    try:
        json_start = response.content.index('{')
        json_end = response.content.rindex('}') + 1
        json_content = response.content[json_start:json_end]
        response_data = json.loads(json_content)
        print("Model Response:", response_data)
        state['technical_series'] = response_data.get('series', [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        state['technical_series'] = []

    # Generate technical series dynamically
    indicators = TechnicalIndicators(df)
    state['generated_series'] = {}

    for series in state['technical_series']:
        name = series['name'].lower()
        params = series.get('parameters', {})
        if hasattr(indicators, f"calculate_{name}"):
            result = getattr(indicators, f"calculate_{name}")(**params)
            
            # If the result is a dictionary of multiple series, merge into generated_series
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, pd.Series):
                        state['generated_series'][key] = value
            else:
                if isinstance(result, pd.Series):
                    state['generated_series'][name] = result

    essentials = ['Close', 'Open', 'High', 'Low']
    for essential in essentials:
        if essential not in state['generated_series']:
            state['generated_series'][essential] = getattr(indicators, f"calculate_{essential.lower()}")()


    print("Generated Series:", state['generated_series'].keys())  # Debugging output
    
    return state

# Step 3: Determine entry and exit conditions
def determine_conditions(state: State):
    """
    Use the LLM to determine entry and exit conditions based on the query.
    """
    prompt = f"""
    Based on the user's query: "{state['user_query']}" and the generated technical series,
    determine the entry and exit conditions. The available series are: {list(state['generated_series'].keys())}.
    Represent these conditions as JSON objects, like:
    {{
        "entry_conditions": [
            {{"series": "MACD", "operator": ">", "compare_to": "Signal"}}
        ],
        "exit_conditions": [
            {{"series": "RSI", "operator": ">", "compare_to": 70}}
        ]
    }}
    """
    response = model.invoke([HumanMessage(content=prompt)])
    # Parse the JSON content from the AIMessage
    try:
        json_start = response.content.index('{')
        json_end = response.content.rindex('}') + 1
        json_content = response.content[json_start:json_end]
        response_data = json.loads(json_content)
        print("Parsed JSON Response:", response_data)
        state['entry_conditions'] = response_data.get('entry_conditions', [])
        state['exit_conditions'] = response_data.get('exit_conditions', [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        state['entry_conditions'] = []
        state['exit_conditions'] = []
    
    return state

# Step 4: Initialize and run the strategy
def execute_strategy(state: State):
    """
    Generate the required series, create conditions, and backtest the strategy.
    """
    # Map series names dynamically for entry and exit conditions
    def map_condition(cond):
        series_a = state['generated_series'].get(cond['series'])
        if series_a is None:
            raise ValueError(f"Series '{cond['series']}' is not calculated.")
        compare_to = state['generated_series'].get(cond['compare_to'], cond['compare_to'])
        return Condition(series_a=series_a, operator=cond['operator'], series_b=compare_to)

    # Dynamically create entry and exit conditions
    entry_conditions = [map_condition(cond) for cond in state['entry_conditions']]
    exit_conditions = [map_condition(cond) for cond in state['exit_conditions']]

    # Initialize and run the strategy
    strategy = Strategy(data=df, entry_conditions=entry_conditions, exit_conditions=exit_conditions)
    state['strategy_results'] = strategy.main()
    return state

def backtest_strategy(user_query: str):
    # Initialize state
    initial_state = State(
        user_query=user_query,
        technical_series={},
        entry_conditions=[],
        exit_conditions=[],
        strategy_results={},
        generated_series={}
    )

    # Step-by-step execution of the workflow
    state = determine_technical_series(initial_state)
    state = determine_conditions(state)
    state = execute_strategy(state)

    # Display the results
    print("Backtesting Results:")
    print(state['strategy_results'])

    return state['strategy_results']


# Streamlit UI
st.title("Trading Strategy Backtesting Results")

# Input Section
st.sidebar.header("Backtesting Input")
user_query = st.sidebar.text_input("Enter your trading query:", "What is the percentage of winning using MACD crossing strategy?")

# Button to generate example query
if st.sidebar.button("Generate Example Query"):
    example_query = generate_example_query()
    st.sidebar.text_input("Enter your trading query:", example_query, key="example_query")

# Backtesting Button
if st.sidebar.button("Run Backtesting"):
    # Simulate running the backend
    with st.spinner("Running backtesting..."):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("Determining technical series...")
        state = determine_technical_series(State(
            user_query=user_query,
            technical_series={},
            entry_conditions=[],
            exit_conditions=[],
            strategy_results={},
            generated_series={}
        ))
        progress_bar.progress(33)

        progress_text.text("Determining entry and exit conditions...")
        state = determine_conditions(state)
        progress_bar.progress(66)

        progress_text.text("Executing strategy...")
        state = execute_strategy(state)
        progress_bar.progress(100)

        backtesting_results = state['strategy_results']

    # Display Results
    st.subheader("Backtesting Results")
    st.metric(label="Win Percentage", value=f"{backtesting_results['win_percentage']}%")
    st.metric(label="Total Trades", value=f"{backtesting_results['total_trades']}")
    st.metric(label="Winning Trades", value=f"{backtesting_results['winning_trades']}")
    st.metric(label="Losing Trades", value=f"{backtesting_results['losing_trades']}")

    # Display Equity Curve
    st.subheader("Equity Curve")
    equity_curve = backtesting_results['equity_curve']
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()
    st.pyplot(plt)

    # Display Trades
    st.subheader("Trades")
    trade_entries = backtesting_results['trade_entries']
    trade_exits = backtesting_results['trade_exits']

    # Ensure the lengths of trade_entries and trade_exits are the same
    min_length = min(len(trade_entries), len(trade_exits))
    trade_entries = trade_entries[:min_length]
    trade_exits = trade_exits[:min_length]

    trade_data = {
        "Entry Time": [entry[0] for entry in trade_entries],
        "Entry Price": [entry[1] for entry in trade_entries],
        "Exit Time": [exit[0] for exit in trade_exits],
        "Exit Price": [exit[1] for exit in trade_exits],
    }
    trades_df = pd.DataFrame(trade_data)
    st.dataframe(trades_df)

    # Plot Trades on Price Chart
    st.subheader("Price Chart with Trades")
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['close'], label='Close Price')
    plt.scatter(trades_df['Entry Time'], trades_df['Entry Price'], color='green', marker='^', label='Entry')
    plt.scatter(trades_df['Exit Time'], trades_df['Exit Price'], color='red', marker='v', label='Exit')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Instructions
st.sidebar.info(
    """
    Enter your query in the text box and click "Run Backtesting".
    The results will display on the main panel.
    """
)