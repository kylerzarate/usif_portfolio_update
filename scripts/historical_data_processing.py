#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from datetime import datetime
import pandas_datareader as pdr
import pandas_market_calendars as mcal
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Processes consolidated transactions to filter for active transactions, group with running shares and start/end dates, show current holdings
class TransactionsProcessing:
    def __init__(self, consolidated_transactions_df):
        self.consolidated_transactions_df = consolidated_transactions_df
        self.buy_and_sell_transactions_df = {}
        self.grouped_transactions_df = {}
        self.current_transactions_df = {}
        # "Aggs" is for different groupings to handle tickers in multiple strategies within the same fund
        self.aggs = ['Fund','Strategy'] # Creates groupings for Fund level, disregarding strategies, and then Strategy level, disregarding funds (Not Sector at this point since we get that from WRDS)

    # Returns, running shares and start/end dates to feed into pivot tables later
    def create_buy_and_sell_transactions_df(self):
        # Define keywords for action types to use for changes in shares
        active_transactions = ['Bought', 'Sell', 'Reverse Split', 'Reorganization', 'Stock Split (Increase Shares)']

        for agg in self.aggs:
            # Sort transactions by keywords
            buy_and_sell_transactions_df = self.consolidated_transactions_df[self.consolidated_transactions_df['Action'].isin(active_transactions)]
            # Sort by fund, then symbol, then date, then action so running shares can be calculated easier
            buy_and_sell_transactions_df.sort_values(by=[agg,'Symbol','Date','Action'], inplace=True)

            # Calculate running sum of shares for each fund and symbol over time
            buy_and_sell_transactions_df['Running Shares'] = buy_and_sell_transactions_df.groupby([agg, 'Symbol'])['Quantity'].cumsum()

            # Create 'Start Date' column based on first date that appears in 'Fund' and 'Symbol' grouping
            buy_and_sell_transactions_df['Start Date'] = buy_and_sell_transactions_df.groupby([agg, 'Symbol'])['Date'].transform('first').dt.strftime('%Y-%m-%d')
            # Create 'End Date' column, if 'Running Shares' is 0, then use given date, else just use today's date
            buy_and_sell_transactions_df['End Date'] = np.where(
                buy_and_sell_transactions_df['Running Shares'] == 0,
                buy_and_sell_transactions_df['Date'],
                datetime.today() - pd.Timedelta(days=1)
            )
            buy_and_sell_transactions_df['End Date'] = pd.to_datetime(buy_and_sell_transactions_df['End Date'])
            buy_and_sell_transactions_df['End Date'] = buy_and_sell_transactions_df['End Date'].dt.strftime('%Y-%m-%d')
            self.buy_and_sell_transactions_df[agg] = buy_and_sell_transactions_df
        return self.buy_and_sell_transactions_df

    # Groups buy/sell transactions df to show holdings with most recent running shares count and end date
    def create_grouped_transactions_df(self):
        for agg in self.aggs: # iterate through aggs again to create separate dfs (Fund, Strategy)
            grouped_transactions_df = self.buy_and_sell_transactions_df[agg].groupby(['Symbol', agg]).agg(
                { 'Start Date': 'first', # first start date
                  'End Date': 'last', # most recent end date
                  'Running Shares': 'last', # most recent running shares count
                  'Price': 'first' # first to represent initial purchase price
                }
            )
            grouped_transactions_df.reset_index(inplace=True)
            grouped_transactions_df.sort_values(by=[agg,'Symbol'], inplace=True)
            # NS is a company that got acquired, would have to figure out how this works
            grouped_transactions_df = grouped_transactions_df[grouped_transactions_df['Symbol'] != 'NS']

            self.grouped_transactions_df[agg] = grouped_transactions_df
        return self.grouped_transactions_df

    # Filters grouped transactions df for only current holdings
    def create_current_transactions_df(self):
        for agg in self.aggs: # iterate through aggs again for separate dfs (Fund, Strategy)
            group = self.grouped_transactions_df[agg]
            self.current_transactions_df[agg] = group[group['Running Shares'] > 0] # Filter for running shares greater than 0 (handles negatives as well)
        return self.current_transactions_df


# Calculates cash flow from consolidated transactions and passive transactions (fees, interest, dividends, etc.)
class CashCalculations:
    def __init__(self, consolidated_transactions_df):
        self.consolidated_transactions_df = consolidated_transactions_df
        self.total_misc_fees_df = None
        self.total_fees_and_comm_df = None
        self.total_fund_fees = {}
        self.fund_cash_flows = {}
        self.total_cash_flows = None

    # Pulls non-trading Actions for fees from consolidated transactions df
    def create_total_misc_fees_df(self):
        # Creates list for 'Fee|MISC' filter, which grabs any Action that contains 'MISC' or 'Fee', there's a bunch of different transaction names
        fees = list(set(self.consolidated_transactions_df['Action'][self.consolidated_transactions_df['Action'].str.contains('Fee|MISC', na=False)]))
        misc_fees_df = self.consolidated_transactions_df[self.consolidated_transactions_df['Action'].isin(fees)].copy() # 

        # Ensure Amount is negative since these are cash outflows
        misc_fees_df.loc[:, 'Amount'] = -abs(misc_fees_df['Amount'])

        # Group by Fund and Date, then sum amounts
        self.total_misc_fees_df = misc_fees_df.groupby(['Fund', 'Date'])[['Amount']].sum()
        return self.total_misc_fees_df

    # Calculates fees associated with active trading
    def create_total_fees_df(self):
        if self.total_misc_fees_df is None:
            self.create_total_misc_fees_df() # pull non-trading fees first

        temp_df = self.consolidated_transactions_df.copy()
        temp_df['Total Fees & Comm'] = temp_df['Fees & Comm'] * temp_df['Quantity'] # 'Fees & Comm' column is cost per share -> multiply by qty

        # Group by Fund and Date, summing up total fees
        total_fees_and_comm_df = temp_df.groupby(['Fund', 'Date'])[['Total Fees & Comm']].sum()
        total_fees_and_comm_df = total_fees_and_comm_df[total_fees_and_comm_df['Total Fees & Comm'] != 0] # Filter our zero rows

        # Join with total_misc_fees_df and sum the fees
        total_fees_and_comm_df = total_fees_and_comm_df.join(self.total_misc_fees_df, how='outer').fillna(0)
        total_fees_and_comm_df['Total Fees'] = total_fees_and_comm_df['Total Fees & Comm'] + total_fees_and_comm_df['Amount']

        # Drop unnecessary columns
        total_fees_and_comm_df.drop(columns=['Total Fees & Comm', 'Amount'], inplace=True)

        self.total_fees_and_comm_df = total_fees_and_comm_df
        return self.total_fees_and_comm_df

    # Gets total fees for each fund
    def create_fund_total_fees(self):
        if self.total_fees_and_comm_df is None:
            self.create_total_fees_df() # pull total fees first

        # Summing up all total fees for each fund
        self.total_fund_fees = self.total_fees_and_comm_df.groupby(level='Fund')['Total Fees'].sum().to_dict()
        return self.total_fund_fees

    # Calculates net cash flows for each fund over time
    def create_cash_flows(self):
        if not self.total_fund_fees:
            self.create_fund_total_fees() # pull fund total fees first
        for fund in self.consolidated_transactions_df['Fund'].unique(): # Iterate through each Fund in consolidated transactions
            fund_transactions = self.consolidated_transactions_df[self.consolidated_transactions_df['Fund'] == fund] # create fund transactions df
            if fund_transactions.empty:
                continue
            fund_cash_flow = fund_transactions.groupby('Date')['Amount'].sum() # Group by date to get sum of Amount over time for fund cash flows
            fund_cash_flow.sort_index(inplace=True)

            # Ensure there are valid dates before creating a range
            if not fund_cash_flow.empty:
                # Create date range so we can join with other dfs easier, 8/18/23 is hard coded and end date subtracts two days to help adjust for WRDS 1 day lag
                all_dates = pd.date_range(start='8-18-2023', end=(datetime.today() - pd.Timedelta(days=1)), freq='D') 
                fund_cash_flow = fund_cash_flow.reindex(all_dates).fillna(0) # Now we have cash flow over time, with zeros representing no cash flow
                # Merge fees into the cash flow data (Weird merging with series and dataframes, could figure out a better way
                fees_series = self.total_fees_and_comm_df.loc[fund]['Total Fees'] if fund in self.total_fees_and_comm_df.index.get_level_values('Fund') else pd.Series()

                # Reindex fees to align with the full date range, filling missing values with 0
                fees_series = fees_series.reindex(fund_cash_flow.index).fillna(0)

                # Subtract fees on their specific dates
                fund_cash_flow -= fees_series
            self.fund_cash_flows[fund] = fund_cash_flow

        return self.fund_cash_flows # return dictionary with funds as keys and cash flows as values

    # Combines cash flows and starting cash amounts to get cash over time for each fund
    def create_total_cash_flows(self, milner_starting_cash, esg_starting_cash, davidson_starting_cash, school_starting_cash):
        # Starting cash amounts hard coded in main.py script when calling this method
        starting_cash = {
            'Milner': milner_starting_cash,
            'ESG': esg_starting_cash,
            'Davidson': davidson_starting_cash,
            'School': school_starting_cash
        }

        total_cash_flows = {} # Store all fund cash 
        for fund, cash_flow in self.fund_cash_flows.items(): # Iterate through fund cash flows dictionary
            total_cash_flows[fund] = starting_cash[fund] + cash_flow.cumsum() # Add starting cash to cumulative sum of cash flow

        total_cash_flows_df = pd.DataFrame(total_cash_flows) # convert total cash flow to dataframe
        self.total_cash_flows = total_cash_flows_df
        return total_cash_flows # return dataframe with each funds cash amount over time

# Pulls historical data from WRDS and cleans/prepares historical prices to be used in pivot tables later
class HistoricalDataProcessing:
    def __init__(self, grouped_transactions_df, wrds_connection):
        self.grouped_transactions_df = grouped_transactions_df # Use grouped transactions df to get start/end dates for each holding
        self.wrds_connection = wrds_connection # Create WRDS object in main.py script
        self.historical_prices = {}
        self.start_date = None
        self.market_data = None
        self.aggs = ['Fund', 'Strategy'] # Create historical prices separately for Fund and Strategy

    # Defines start date as 5 years ago from today's date (Used for beta calculations)
    def get_start_date(self):
        start_date = datetime.today() - pd.Timedelta(days=365*5)
        start_date = start_date.strftime('%Y-%m-%d')
        self.start_date = start_date
        return self.start_date

    # Orchestrates iterating through grouped transactions df rows, WRDS query, and combining/storing price data
    def get_historical_prices(self):
        all_prices = {} # Dictionary to store historical prices once finished
        for agg in self.aggs: # Iterate through 'aggs' to create different groupings (Fund, Strategy)
            category_to_data = {} # Dictionary to store multiple levels in order of category (within agg) then ticker (within category)
            # Iterate through each row of grouped transactions df
            for _, row in self.grouped_transactions_df[agg].iterrows():
                ticker = row['Symbol']
                category = row[agg] # either Fund or Strategy column value
                start_date = self.start_date # Defined 5-years ago start date
                end_date = row['End Date']
                # Creates SQL query to use with WRDS connection object for each row
                price_query = self.price_query(ticker, start_date, end_date)
                try:
                    price_data = self.wrds_connection.raw_sql(price_query) # Sends SQL query to WRDS connection object
                    if not price_data.empty:
                        price_data.rename(columns={'date': 'Date'}, inplace=True)
                        price_data['Date'] = pd.to_datetime(price_data['Date'])
                        if category not in category_to_data:
                            category_to_data[category] = {}
                        category_to_data[category][ticker] = {'data': price_data}
                except Exception as e:
                    print(f"Error fetching data for {ticker} ({category}): {e}")
            # Combines category and ticker dictionaries into one df to be prepared and stored under all_prices
            all_prices[agg] = self.combine_price_data(category_to_data, agg)
        return all_prices # Dictionary containing Fund: fund level prices df, Strategy: strategy level prices df

    # SQL query template to be used in get_historical_prices() with dynamic ticker and start/end dates from grouped transactions df
    """
    Pulls from daily security prices (sec_dprc) table in Compustat (comp_na_daily_all) database within WRDS
    Columns:
    gsector: GICS sectors from S&P
    tpci: security type id - used for identifying ETFs, ADRs, etc.
    prccd:  closing prices (1 day lag since this isn't live data)
    ajexdi: daily adjustment factor for prices to adjust for stock splits, dividends, etc.
    iid: security level id - companies can have multiple securities, so we have to identiy the primary trading security
    gvkey: company level id - used with iid to get unique ids
    Tables:
    prices: historical security trading info (close price, adj factors, issue id)
    company: company info (name, sector, country)
    security: security info (tpci)
    """
    def price_query(self, ticker, start_date, end_date):
        return f"""
            SELECT
                comp.conm AS company_name,
                comp.gsector AS "Sector",
                security.tpci AS tpci,
                comp.loc AS country,
                prices.prccd AS price,
                prices.ajexdi AS adjustment,
                prices.datadate AS date,
                prices.iid AS issue_id
            FROM
                comp_na_daily_all.sec_dprc AS prices
            INNER JOIN
                comp_na_daily_all.company AS comp
            ON
                prices.gvkey = comp.gvkey
            INNER JOIN
                comp_na_daily_all.security AS security
            ON
                prices.gvkey = security.gvkey
            WHERE
                security.tic = '{ticker}'
                AND prices.datadate BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY
                prices.datadate ASC
        """

    # Pulls historical SPY prices from WRDS with similar price query (spy_code = SPY gvkey found on WRDS)
    def get_market_data(self, spy_code, start_date, end_date):
        market_query = f"""
            SELECT
                prices.gvkey,
                prices.datadate,
                prices.prccd AS close_price,
                prices.ajexdi AS adjustment_factor
            FROM
                comp_na_daily_all.sec_dprc AS prices
            WHERE
                prices.gvkey = '{spy_code}'
                AND prices.datadate BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY
                prices.datadate ASC
            """

        market_data = self.wrds_connection.raw_sql(market_query)
        market_data.rename(columns={'datadate': 'Date'}, inplace=True)
        market_data['Date'] = pd.to_datetime(market_data['Date'])
        market_data.set_index('Date', inplace=True)
        market_data['adjprc'] = market_data['close_price'] / market_data['adjustment_factor'] # Adj prices = price / adj factor
        market_data.drop(columns=['close_price', 'adjustment_factor', 'gvkey'], inplace=True)
        market_data.rename(columns={'adjprc': 'SPY'}, inplace=True)

        self.market_data = market_data
        return self.market_data # Return df for historical SPY prices

    # Used in get_historical_prices() to combine category/ticker dictionaries into a dataframe
    def combine_price_data(self, category_to_data, agg):
        all_prices = [] # Create list to flatten/store dictionaries in one place
        for category, tickers in category_to_data.items():
            for ticker, info in tickers.items():
                data = info['data']
                data[agg] = category # agg for Fund, Strategy level tables
                data['Symbol'] = ticker
                all_prices.append(data)
        return pd.concat(all_prices, ignore_index=True) # Return converted dataframe from list

    # Cleans/prepares output from get_historical_prices()
    def prepare_historical_prices(self):
        historical_prices_dict = self.get_historical_prices() # Create get_historical_prices() 
        
        for agg in self.aggs: # Iterate through Fund, Strategy levels
            historical_prices = historical_prices_dict[agg] # Keys for Fund, Strategy
            
            historical_prices['adjprc'] = historical_prices['price'] / historical_prices['adjustment'] # Adj price = price / adj factor
            # Hardcoded GICS to Sector map from S&P
            sector_codes = {
                '10': 'Energy',
                '15': 'Materials',
                '20': 'Industrials',
                '25': 'Cons. Disc.',
                '30': 'Cons. Staples',
                '35': 'Healthcare',
                '40': 'Financials',
                '45': 'Technology',
                '50': 'Comm. Services',
                '55': 'Utilities',
                '60': 'Real Estate'
            }
            
            historical_prices['Sector'] = historical_prices['Sector'].map(sector_codes) # Map GICS codes to Sector 
            # If tpci is '%', fill sector with 'ETF'
            historical_prices.loc[historical_prices['tpci'] == '%', 'Sector'] = 'ETF'
            historical_prices = historical_prices[~historical_prices['issue_id'].str.contains('[a-zA-Z]')] # Filter out security ids with letters (not primary)
            historical_prices['issue_id'] = pd.to_numeric(historical_prices['issue_id']) # Convert issue ids to numeric
            # Identify and isolate 'F' type securities (code for ADRS that have a different primary issue id than non ADR securities)
            historical_prices_adr = historical_prices[historical_prices['tpci'] == 'F']
            # Filter out ADR rows where tpci is F in main historical prices df
            historical_prices = historical_prices[historical_prices['tpci'] != 'F']
            # In main historical prices df, keep only the lowest issue id for each ticker in original dataframe (iid closest to 1 = primary security)
            historical_prices = historical_prices.sort_values('issue_id').groupby([agg, 'Symbol', 'Date']).first().reset_index()
            # In ADR historical prices df, keep only rows where issue id is 90 from pulled dataframe (ADR way of identifying primary securities)
            historical_prices_adr = historical_prices_adr[historical_prices_adr['issue_id'] == 90]
            # Recombine ADR and main historical prices dfs
            historical_prices = pd.concat([historical_prices, historical_prices_adr])

            self.historical_prices[agg] = historical_prices
        return self.historical_prices # Return dictionary of prepared historical prices at Fund, Strategy level

# Method to update transaction dfs (buy_and_sell_transactions_df, grouped_transactions_df) with sector values
def update_transactions_df(transactions_df, historical_prices, aggs):
    for agg in aggs: # Iterates through the dictionary keys for Fund, Strategy dfs
       prices = historical_prices[agg]
       transactions_df_agg = transactions_df[agg]
       unique_symbols_historical_prices = prices[[agg, 'Symbol', 'Sector']].drop_duplicates() # Gets unique agg, Symbol, and Sector rows
       transactions_df_agg = transactions_df_agg.merge(unique_symbols_historical_prices, on=[agg, 'Symbol'], how='left') # Use unique agg and Symbols to join Sector rows
       transactions_df[agg] = transactions_df_agg.dropna(subset=['Sector'])

    return transactions_df # Return updated Sector version of transactions df


# # Create equity value pivot tables with dates as rows, prices * shares as values, and tuple of agg (Fund, Strategy) and ticker as the columns
class CreatePivotTables:
    def __init__(self, buy_and_sell_transactions_df, historical_prices):
        self.buy_and_sell_transactions_df = buy_and_sell_transactions_df
        self.historical_prices = historical_prices
        self.pivot_tables = {} # Stored equity value pivot tables for calculating totals/weights/performance metrics w/ 8/18/23 to today data
        self.price_pivot_table = {} # Stored price pivot tables for calculating betas with 5 years of past returns
        self.levels = ['Fund', 'Sector', 'Strategy'] # Now that we have WRDS Sectors, we can start grouping by Sector as well

    # Calculates equity values by multipling pivot tables of prices and running shares over time for each level (Fund, Sector, Strategy)
    def create_equity_pivot_tables(self):          
        for level in self.levels: # Iterates through levels (Fund, Sector, Strategy) to grab data based on defined columns
            if level == 'Sector': # Since there isn't a Sector transactions or historical prices df, pull from Fund (a bit more rows than Strategy)
                prices = self.historical_prices['Fund']
                transactions = self.buy_and_sell_transactions_df['Fund']
            else: # Else just pull from dictionary based on level/agg key
                prices = self.historical_prices[level]
                transactions = self.buy_and_sell_transactions_df[level]
            # Create price pivot table with date as rows, level/Symbol tuple as columns, and adj prices as values
            price_pivot_table = prices.pivot_table(index='Date', columns=[level, 'Symbol'], values='adjprc')
            price_pivot_table = price_pivot_table.sort_index(axis=1, level=0)
            
            # Reindex to include weekend dates - WRDS prices only includes trading days for closing prices
            all_dates = pd.date_range(start=price_pivot_table.index.min(), end=price_pivot_table.index.max(), freq='D')
            price_pivot_table = price_pivot_table.reindex(all_dates) # Reindex pivot table with all dates
            # Forward-fill the missing values, using Friday prices for Saturday and Sunday
            price_pivot_table = price_pivot_table.ffill()
            price_pivot_table.index = price_pivot_table.index.tz_localize(None) # Remove timezones from dates (might have to use dt.date instead)
            
            # Create pivot table with date as rows, level/Symbol tuple as columns, and running counts of shares as values (last running shares count for each date)
            running_shares_pivot_table = transactions.pivot_table(index='Date', columns=[level, 'Symbol'], values='Running Shares', aggfunc='last')
            # Expand dates with all_dates to match price pivot table for easier matching
            running_shares_pivot_table = running_shares_pivot_table.reindex(all_dates).fillna(method='ffill')
            running_shares_pivot_table = running_shares_pivot_table.fillna(0)
            # Calculate values by multiplying matching row/column values from price and running shares pivot tables
            equity_value_pivot_table = price_pivot_table * running_shares_pivot_table
            self.pivot_tables[level] = equity_value_pivot_table
            self.price_pivot_table[level] = price_pivot_table

        return self.pivot_tables, self.price_pivot_table # Return two dictionaries, each containing subsets of levels dictionaries (Fund, Sector, Strategy)


# Calculates total values, handles cash columns, calculates portfolio/category weights
class TotalsAndWeights:
    def __init__(self, equity_pivot_tables, total_cash_flows):
        self.equity_pivot_tables = equity_pivot_tables  # Dictionary of category-level pivot tables
        self.total_cash_flows = total_cash_flows  # Only applies to fund level (For AUM table)
        self.category_totals = {}
        self.category_weights = {}
        self.individual_weights = {}
        self.weights_within_categories = {}
        self.fund_values = {}
        
        # Total Cash for Strategy and Sector tables (we don't need fund cash for these)
        self.total_portfolio_cash = pd.DataFrame(total_cash_flows).sum(axis=1).rename('Total Cash')
        
    # Calculates totals for each category by grouping by first level in column (individual fund/sector/strategies, not tickers)
    def calculate_category_equity_values(self):
        for level, pivot_table in self.equity_pivot_tables.items():
            # Sum across tickers for each category
            category_totals = pivot_table.groupby(level=0, axis=1).sum()
            self.category_totals[level] = category_totals
        return self.category_totals
        
    # Include cash columns in weightings and beta calculations based on Fund or Sector/Strategy tables
    def add_cash_columns(self):
        fund_cash_amounts_df = pd.DataFrame(self.total_cash_flows) # Grab for individual fund cash amounts
        total_cash_amount = self.total_portfolio_cash # Grab for total cash amounts

        for level, pivot_table in self.equity_pivot_tables.items():
            if level == 'Fund':
                for fund in fund_cash_amounts_df.columns: # Fund table should have individual fund cash amounts
                    pivot_table[(fund, f'{fund} Cash')] = fund_cash_amounts_df[fund]
            elif level != 'Fund': # Sector/Strategy tables should just have a total cash amount
                pivot_table[('Total Cash', 'Cash')] = total_cash_amount

            self.equity_pivot_tables[level] = pivot_table
        return self.equity_pivot_tables # Return adjusted cash versions of dictionary

    # Sums fund equity value totals and fund cash amounts to get fund total values for AUM table
    def total_fund_values_aum(self):
        if 'Fund' in self.category_totals: # Get Fund level category equity values
            fund_equity = self.category_totals['Fund']
            # Convert the cash flows dictionary to a DataFrame
            cash_flows_df = pd.DataFrame(self.total_cash_flows)
            # Reindex to ensure both dataframes have the same dates and funds (if necessary)
            aligned_cash_flows = cash_flows_df.reindex_like(fund_equity)
            # Add the cash flow values to the equity values, aligned by the index (dates) and columns (funds)
            adjusted_fund_totals = fund_equity.add(aligned_cash_flows, axis=0, fill_value=0)
            # Assign the adjusted totals back to the category_totals
            self.fund_values['Fund'] = adjusted_fund_totals
            self.fund_values['Fund']['Portfolio'] = adjusted_fund_totals.sum(axis=1) # Sum across fund totals to get Portfolio total
            # Create an extra key for Portfolio total
            portfolio_df = pd.DataFrame(self.fund_values['Fund']['Portfolio'])
            portfolio_df.columns = ['Portfolio']
            self.category_totals['Portfolio'] = portfolio_df # Add Portfolio key to category totals dictionary (helps with PerformanceMetrics script)
        return self.fund_values # Returns dictionary with fund totals

    # Calculates each tickers weight within the entire portfolio
    def calculate_individual_weights(self):
        for level, pivot_table in self.equity_pivot_tables.items():
            # Divide each ticker equity value by sum of entire row
            self.individual_weights[level] = pivot_table.div(pivot_table.sum(axis=1), axis=0).fillna(0)
        return self.individual_weights

    # Calculates each categories weights
    def calculate_category_weights(self):
        for level, pivot_table in self.individual_weights.items():
            # Sums individual weights based on upper level category in columns
            self.category_weights[level] = pivot_table.groupby(level=0, axis=1).sum().fillna(0)
        return self.category_weights

    # Calculates weights of each ticker within a given category
    def calculate_weights_within_categories(self):
        for level, pivot_table in self.equity_pivot_tables.items():
            category_total_df = self.category_totals.get(level) # Filters columns for given category
            self.weights_within_categories[level] = pivot_table.div(category_total_df, level=0).fillna(0) # Divide each ticker by category total
        return self.weights_within_categories

# Used to set up inputs for performance_metrics classes
class PerformanceMetricsSetup():
    def __init__(self, aytd, t_bill_code):
        self.calendar_aytd = aytd # aytd = Academic-Year-To-Date (SIF start date: 8/19/2024 for 2024-2025 cohort)
        self.t_bill_code = t_bill_code # t bill code from FRED
        self.end_date = datetime.today() - pd.Timedelta(days=1)
        self.risk_free_rate = None
        self.trading_aytd_length = None
        self.calender_aytd_length = None

    # Gets number of calendar days from aytd to today's date (includes weekends/holidays)
    def calculate_calendar_aytd_length(self):
        aytd_date = self.calendar_aytd
        end_date = self.end_date
        aytd_length = (end_date - datetime.strptime(aytd_date, '%Y-%m-%d')).days
        self.calendar_aytd_length = aytd_length
        return self.calendar_aytd_length

    # Gets number of trading days from aytd to today's date (excludes non-trading days)
    def calculate_trading_aytd_length(self):
        trading_aytd = self.trading_aytd_length
        aytd_date = self.calendar_aytd
        end_date = self.end_date

        nyse = mcal.get_calendar('NYSE')  # Gets trading day calendar from NYSE using pandas_market_calendars
        trading_days = nyse.valid_days(start_date=aytd_date, end_date=end_date) # Gets valid trading days based on NYSE
        trading_aytd_length = len(trading_days)
        self.trading_aytd_length = trading_aytd_length
        return self.trading_aytd_length

    # Gets most Treasury Bill data for risk-free rate of return
    def get_risk_free_rate(self):
        t_bill_code = self.t_bill_code
        t_bill_data = pdr.DataReader(t_bill_code, 'fred') # Pulls t-bill data from FRED
        risk_free_rate = t_bill_data.iloc[-1][t_bill_code] / 100 # return most recent value
        risk_free_rate = (1 + risk_free_rate) ** (1/365) - 1 # Scaling risk-free rate to daily rate
        self.risk_free_rate = risk_free_rate
        return self.risk_free_rate

