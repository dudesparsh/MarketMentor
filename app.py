import json
import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from dotenv import load_dotenv
import os

from openai import OpenAI

# Sample example
# What is the  Blackrock fund with high profitability of more than 10%. I am looking for short term of 3 months

# What is the current stock price of apple 

# Can you plot the stock price of apple

# Can you tell what is best blackrock fund for low risk and high profit in past 5 years. I am looking for 10% profit
load_dotenv()

# Now you can access the environment variables
api_key = os.getenv("OPENAI_API_KEY")
api_org = os.getenv("OPENAI_ORG")

client = OpenAI(api_key=api_key, organization=api_org)

def getStockPrice(ticker):
    """
    Fetches the latest closing stock price for a given ticker symbol over the last year.
    Returns: The most recent closing price as a string.

    """
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

def calculateSMA(ticker, window):
    """
    Computes the Exponential Moving Average (EMA) of the stock's closing price over the specified window (number of days).
    Returns: The latest EMA value as a string.

    """
    
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculateEMA(ticker, window):
    """
    
    """
    
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculateRSI(ticker):
    """
    Calculates the Relative Strength Index (RSI) for the stock's closing price over the last year to measure its momentum.
    Returns: The most recent RSI value as a string.

    """
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up/ema_down
    return str(100 - (100/(1+rs)).iloc[-1])

def calculateMACD(ticker):
    """
    Calculates the Moving Average Convergence Divergence (MACD) of the stock, including its signal line and histogram, to assess trend strength.
    Returns: A string with the latest values for MACD, the signal line, and the MACD histogram. 
    """
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()

    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal

    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'

def plotStockPrice(ticker):
    """
    Plots and saves a graph of the stock’s closing price over the past year.
    Returns: The plot is saved as 'stock.png', and no direct return value from the function.
    """
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data.Close)
    plt.title('{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()


def Blackrock_optimizer(detail_prompt):


    data = {'Blackrock Growth Fund': {'1_month': {'return': 2.47, 'volatility': 17.36}, '3_months': {'return': 8.69, 'volatility': 22.74}, '6_months': {'return': -10.56, 'volatility': 21.62}, '1_year': {'return': -9.68, 'volatility': 15.66}, '3_years': {'return': 16.56, 'volatility': 17.06}, '5_years': {'return': -0.42, 'volatility': 17.13}}, 'Blackrock Balanced Fund': {'1_month': {'return': 1.71, 'volatility': 17.84}, '3_months': {'return': 8.52, 'volatility': 24.25}, '6_months': {'return': -13.27, 'volatility': 20.91}, '1_year': {'return': 19.73, 'volatility': 12.31}, '3_years': {'return': -1.96, 'volatility': 6.48}, '5_years': {'return': -13.65, 'volatility': 8.59}}, 'Blackrock Equity Index': {'1_month': {'return': -0.12, 'volatility': 7.17}, '3_months': {'return': -2.29, 'volatility': 21.95}, '6_months': {'return': -6.2, 'volatility': 17.92}, '1_year': {'return': -16.24, 'volatility': 5.05}, '3_years': {'return': -13.14, 'volatility': 24.65}, '5_years': {'return': -34.06, 'volatility': 24.76}}, 'Blackrock Global Fund': {'1_month': {'return': -0.16, 'volatility': 15.91}, '3_months': {'return': 0.68, 'volatility': 9.16}, '6_months': {'return': -7.47, 'volatility': 13.3}, '1_year': {'return': 1.35, 'volatility': 7.51}, '3_years': {'return': 13.11, 'volatility': 17.55}, '5_years': {'return': 28.76, 'volatility': 6.42}}, 'Blackrock Tech Fund': {'1_month': {'return': -1.87, 'volatility': 20.29}, '3_months': {'return': 9.84, 'volatility': 20.29}, '6_months': {'return': 2.22, 'volatility': 22.46}, '1_year': {'return': 0.19, 'volatility': 8.59}, '3_years': {'return': -9.29, 'volatility': 19.99}, '5_years': {'return': 12.65, 'volatility': 11.17}}, 'Blackrock Healthcare Fund': {'1_month': {'return': -0.69, 'volatility': 8.2}, '3_months': {'return': -7.3, 'volatility': 19.13}, '6_months': {'return': 8.83, 'volatility': 16.23}, '1_year': {'return': 1.95, 'volatility': 18.21}, '3_years': {'return': 3.38, 'volatility': 5.87}, '5_years': {'return': 39.58, 'volatility': 11.08}}, 'Blackrock Dividend Fund': {'1_month': {'return': 3.63, 'volatility': 24.05}, '3_months': {'return': 6.35, 'volatility': 8.35}, '6_months': {'return': 6.18, 'volatility': 12.62}, '1_year': {'return': -0.87, 'volatility': 12.85}, '3_years': {'return': -24.35, 'volatility': 8.17}, '5_years': {'return': 7.46, 'volatility': 20.49}}, 'Blackrock Energy Fund': {'1_month': {'return': -1.66, 'volatility': 10.77}, '3_months': {'return': -2.36, 'volatility': 5.09}, '6_months': {'return': -11.85, 'volatility': 14.5}, '1_year': {'return': 18.7, 'volatility': 23.35}, '3_years': {'return': -4.98, 'volatility': 15.2}, '5_years': {'return': 34.32, 'volatility': 22.17}}, 'Blackrock Infrastructure Fund': {'1_month': {'return': -2.4, 'volatility': 8.7}, '3_months': {'return': -2.67, 'volatility': 20.73}, '6_months': {'return': 4.49, 'volatility': 6.61}, '1_year': {'return': -7.4, 'volatility': 8.35}, '3_years': {'return': -18.7, 'volatility': 19.09}, '5_years': {'return': -1.27, 'volatility': 15.85}}, 'Blackrock Consumer Fund': {'1_month': {'return': -0.97, 'volatility': 12.41}, '3_months': {'return': -1.57, 'volatility': 20.74}, '6_months': {'return': -4.4, 'volatility': 20.93}, '1_year': {'return': -10.87, 'volatility': 11.74}, '3_years': {'return': -20.24, 'volatility': 24.36}, '5_years': {'return': -38.42, 'volatility': 5.29}}}
    pass
    
    
    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{"role": "system", "content": """

    Suggest the best mutual fund based on user preferences.

    Parameters:
    - data: JSON-like dictionary containing mutual fund data.
    - timeframe: The period to compare (e.g., '1_month', '1_year').
    - preference: 'profit' (default) or 'volatility'. Determines whether to suggest based on highest return or lowest volatility.
    
    Returns:
    - Suggested fund name and its corresponding data.


"""},
    {"role": "user", "content": f" The data is {data}. Further the timeframe and preferences are {detail_prompt}"}])

    return response

    # Sample data
    # timeframe = "3_months"
    # profit = "15%"
    # volatility = "High of above 10%"
    # print(Blackrock_optimizer(profit, timeframe, volatility).choices[0].message.content)


def Blackrock_group(prompt):
    # Blackrock_optimizer

    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[{"role": "system", "content": """

    Extract these details from the user prompt.

    - timeframe: The period to compare (e.g., '1_month', '1_year').
    - profit: The required profit (by default 10%)
    - volatility: The required volatility (by default 5%)
    

"""},
    {"role": "user", "content": prompt}])

    return response



import re

functions = [
    {
        'name':'getStockPrice',
        'description':'Gets the latest stock price given the ticker symbol of a company.',
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                }
            },
            'required': ['ticker']
        }
    },
    {
        'name':'calculateSMA',
        'description':'Calculate the simple moving average for a given stock ticker and a window.',
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the SMA'
                }
            },
            'required': ['ticker', 'window'],
        },

    },
    {
        'name':'calculateEMA',
        'description':'Calculate the exponential moving average for a given stock ticker and a window.',
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                },
                'window': {
                    'type': 'integer',
                    'description': 'The timeframe to consider when calculating the EMA'
                }
            },
            'required': ['ticker', 'window'],
        },
    },
    {
        'name':'calculateRSI',
        'description':'Calculate the RSI for a given stock ticker.',
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                }
            },
            'required': ['ticker'],
        },
    },
    {
        'name':'calculateMACD',
        'description':'Calculate the MACD for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                },
            },
            'required': ['ticker'],
        },
    },
    {
        'name':'plotStockPrice',
        'description':'Plot the stock price for the last year givent the ticker symbol of a company',
        'parameters':{
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple).'
                    
                }
            },
            'required': ['ticker'],
        },
    }
]

availableFunctions = {
    'getStockPrice': getStockPrice,
    'calculateSMA': calculateSMA,
    'calculateEMA': calculateEMA,
    'calculateRSI': calculateRSI,
    'calculateSMACD': calculateMACD,
    'plotStockPrice': plotStockPrice,

}

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('MarketMentor: Fund Analysis Assistant')

user_input = st.text_input('Your input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': f'{user_input}'})

        pattern = 'Blackrock'
        # string = "Blackrock"
        if(re.search(pattern, user_input, re.IGNORECASE)):
            details = Blackrock_group(user_input)
            
            response = Blackrock_optimizer(details.choices[0].message.content)
            response_message = response.choices[0].message

            print(response_message)
            st.text(response_message.content)
            st.session_state['messages'].append({'role': 'assistant', 'content': response_message.content})

        else:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=st.session_state['messages'],
                functions=functions,
                function_call='auto'
            )

            print(response)

            response_message = response.choices[0].message

            if response_message.function_call:
                function_name = response_message.function_call.name
                function_args = json.loads(response_message.function_call.arguments)

                if function_name in ['getStockPrice', 'calculateRSI', 'calculateMACD', 'plotStockPrice']:
                    args_dict = {'ticker': function_args.get('ticker')}
                elif function_name in ['calculateSMA', 'calculateEMA']:
                    args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}
                
                function_to_call = availableFunctions[function_name]
                function_response = function_to_call(**args_dict)

                if function_name == 'plotStockPrice':
                    st.image('stock.png')
                else:
                    st.session_state['messages'].append(response_message)
                    st.session_state['messages'].append(
                        {
                            'role': 'function',
                            'name': function_name,
                            'content': function_response
                        }
                    )
                    second_response = client.chat.completions.create(
                        model='gpt-3.5-turbo',
                        messages=st.session_state['messages']
                    )
                    st.text(second_response.choices[0].message.content)
                    st.session_state['messages'].append({'role': 'assistant', 'content': second_response.choices[0].message.content})
            else: 
                st.text(response_message.content)
                st.session_state['messages'].append({'role': 'assistant', 'content': response_message.content})
    except Exception as e:
        raise e