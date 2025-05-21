import pandas as pd
from datetime import datetime
import numpy as np
import pandas_market_calendars as mcal
import warnings
warnings.filterwarnings("ignore")

# Prepares returns for each ticker based on equity or price pivot table and window of returns to be used in other classes
# Windows (trading days): 1 for daily, 5 for weekly, 21 for monthly, 252 for yearly
class ReturnPreparer:
    @staticmethod
    def prepare_returns(pivot_table, sp500_data, window): # Specify 21 day window for beta func, 1 day window for perf.metrics
        end_date = sp500_data.index.max() # get start and end dates using max and min of dates
        start_date = sp500_data.index.min()
        nyse = mcal.get_calendar('NYSE') # Get NYSE trading calendar

        # Get all valid trading days between the start and end dates
        trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)
        trading_days = trading_days.tz_localize(None) # Remove timestamps

        # Filter to include only trading days
        df_trading_days = sp500_data[sp500_data.index.isin(trading_days)]

        # Select dates at n-trading-day intervals (21 for beta, 1 for performance metrics)
        selected_dates = df_trading_days.index[::-window]  # Reverse the index and take every 21st day [start:end:step]
        sp500_returns = df_trading_days.loc[selected_dates] # Retrieve only values from selected trading dates
        sp500_returns.sort_index(inplace=True) # Sort back to oldest to most recent

        # Calculate percentage changes between these selected dates
        sp500_returns = sp500_returns.pct_change()
        sp500_returns.dropna(inplace=True)
        sp500_returns = sp500_returns.squeeze() # Convert to a Series to make covariance calculation easier

        portfolio_returns = {}
        # Iterate through levels of pivot tables (fund, sector, strategy)
        for level, df in pivot_table.items():
            portfolio_returns[level] = {} # Create dictionary within dictionary
            df = df[df.index.isin(selected_dates)] # Match index with selected dates
            portfolio_returns[level] = df.pct_change() # Calculate returns for pivot table
            portfolio_returns[level] = portfolio_returns[level].iloc[1:] # Remove first row (null)
            portfolio_returns[level] = portfolio_returns[level].replace([np.inf, -np.inf], np.nan)

        return portfolio_returns, sp500_returns # Return dictionary of portfolio returns and market returns series

# Use 5 years of monthly returns to calculate betas for individual tickers and weighted betas using price pivot table and  different weights pivot tables
class Betas:
    def __init__(self, price_pivot_table, sp500_data, individual_weights, weights_within_categories, category_weights):
        self.price_pivot_table = price_pivot_table # Used to load ticker prices into ReturnsPreparer class
        self.sp500_data = sp500_data # Use to load SPY prices into ReturnsPreparer class
        self.individual_weights = individual_weights
        self.weights_within_categories = weights_within_categories
        self.category_weights = category_weights
        self.current_weights = {}
        self.betas = {}
        # Use ReturnsPreparer to get 5 years of monthly returns using price pivot table and SPY prices
        self.portfolio_returns, self.sp500_returns = ReturnPreparer.prepare_returns(self.price_pivot_table, self.sp500_data, 21) # set window to 21-day intervals for monthly returns
        self.total_weighted_betas = {}
        self.weighted_fund_betas = {}
    
    # Calculate individual ticker betas
    def calculate_beta(self):
        betas = {}
        # Get prepared dictionary of portfolio returns and individual df of sp500 returns
        portfolio_returns, sp500_returns = self.portfolio_returns, self.sp500_returns

        # Create cash columns with zero values in portfolio_returns (to get betas of zero and not affect other metrics)
        for level, df in portfolio_returns.items():
            if level == 'Fund': # If fund, add cash column for each fund
                for fund, categories in df.items():
                    df[(fund[0], f'{fund[0]} Cash')] = 0 # Cash has beta of 0
            elif level != 'Fund': # if not fund (sector, strategy), add one total cash column
                df[('Total Cash', 'Cash')] = 0 # Cash has beta of 0

        # Iterate through each level of pivot tables (fund, sector, strategy)
        for level, df in portfolio_returns.items():
            # Create dictionary within dictionary
            betas[level] = {}
            # Iterate through each column of pivot table
            for category in df.columns:  # df.columns is a list of column names
                if 'Cash' in category: # Set to zero if 'Cash' in column name - just create new cash columns in price pivot tables bec we use price and not equity pivot tables
                    betas[level][category] = 0 # Cash has beta of 0
                covariance = df[category].cov(sp500_returns)
                variance = sp500_returns.var()
                # Beta formula = covariance(portfolio returns, benchmark returns) / variance(benchmark returns)
                betas[level][category] = covariance / variance # Could also use OLS regression but that sounds more complicated to implement
        self.betas = betas
        return betas # Returns dictionary of individual ticker betas stored in subsets of level and category dictionaries

    # Returns last row of weights pivot table to get current weights
    def find_current_weights(self, weights):
        current_weights = {}
        for level, categories in weights.items():
            current_weights[level] = {}
            for category, weight_df in categories.items():
                current_weights[level][category] = weight_df.iloc[-1] # Store only last row values
        return current_weights

    # Multiplies current weights by betas to get weighted betas for each ticker within each category (not entire portfolio)
    def calculate_individual_weighted_betas(self):
        weighted_category_betas = {}
        current_individual_weights = self.find_current_weights(self.weights_within_categories) # use find_current_weights() method to get last row
        for level, categories in self.betas.items():

            weighted_category_betas[level] = {}
            for category, beta in categories.items():
                weighted_category_betas[level][category] = beta * current_individual_weights[level][category]

        return weighted_category_betas

    # Calculate category betas using weighted average betas of tickers within category
    def calculate_category_total_weighted_betas(self):
        weighted_betas = self.calculate_individual_weighted_betas() # Calculate individual weighted betas first with calculate_individual_weighted_betas()
        total_weighted_beta = {}
        for level, categories in weighted_betas.items():
            total_weighted_beta[level] = {}
            for category, weighted_beta in categories.items():
                # Pull the upper-level category (categories within Fund, Sector, Strategy) and assign the weighted beta
                upper_category = category[0]  # Assumes the upper-level category is the first element in the tuple
                if upper_category not in total_weighted_beta[level]:
                    total_weighted_beta[level][upper_category] = 0 # Used for cash
                # Add the weighted betas to the upper-level category
                total_weighted_beta[level][upper_category] += weighted_beta

        self.total_weighted_betas = total_weighted_beta
        return total_weighted_beta # Return dictionary of category betas for each level

    # Calculates portfolio beta by summing Fund-level category weighted betas
    def calculate_portfolio_total_weighted_beta(self):
        weighted_fund_betas = {}
        # Get current category weights using current weights function
        current_category_weights = self.find_current_weights(self.category_weights)
        # Iterate through category weighted betas
        for level, categories in self.total_weighted_betas.items():
            if level == 'Fund': # Only need fund betas and weights to get portfolio beta
                for category, beta in categories.items():
                    weighted_fund_betas[category] = beta * current_category_weights[level][category]

        self.weighted_fund_betas = weighted_fund_betas
        portfolio_beta = sum(weighted_fund_betas.values()) # Sum weighted fund betas to get portfolio betas
        self.total_weighted_betas['Portfolio'] = {} # Store in 'Portfolio' subset of dictionary
        self.total_weighted_betas['Portfolio']['Portfolio'] = portfolio_beta # Used to get to same level as other betas (makes alpha calculation easier)
        return weighted_fund_betas

# Used to calculate various performance metrics using returns, category values, betas - should be able to add new metrics more easily
class PerformanceMetrics:
    def __init__(self, category_values, betas, sp500_data, aytd, risk_free_rate):
         # Add new subset in dictionary to calculate performance metrics for market data as well
        category_values['Benchmark'] = sp500_data # Assign before initializing category values
        
        # Use category values instead of equity pivot tables (we don't need individual ticker performance metrics at the moment - could be something to look into)
        self.category_values = category_values
        self.sp500_data = sp500_data
        self.betas = betas
        
        
        # Inject market beta of 1 to avoid missing values
        if 'Benchmark' not in self.betas:
            self.betas['Benchmark'] = {'SPY': 1}
            
        self.aytd = aytd # Should be aytd_trading_length from class input
        self.risk_free_rate = risk_free_rate
        
        # Define number of trading days for different intervals for sampling the returns data
        self.intervals = {
            'Weekly': 5,
            'Monthly': 21,
            'AYTD': aytd, # Uses trading aytd length
            'Yearly': 252
        }
        # Call ReturnsPreparer with category values and benchmark data, with window set to 1 for daily returns
        self.portfolio_returns, self.sp500_returns = ReturnPreparer.prepare_returns(self.category_values, self.sp500_data, 1)
        
        # Dictionaries for storing performance metrics, add new dictionaries for new metrics after creating method
        self.sharpe_ratios = {}
        self.sortino_ratios = {}
        self.information_ratios = {}
        self.max_drawdowns = {}
        self.alpha_values = {}
        self.standard_deviations = {}
        self.average_return = {}
        self.cumulative_returns = {}
        self.interval_betas = {}
        self.interval_r_square = {}
        self.treynor_ratios = {}

    # Use intervals dictionary to get different samples of daily returns
    def interval_returns(self):
        interval_returns = {}
        # Iterate through intervals dictionary, retrieving the number of days sampled for each key
        for interval, days in self.intervals.items():
            interval_returns[interval] = {}
            for level, df in self.portfolio_returns.items(): # Iterate through portfolio returns
                # Get sample of most recent rows using 'days' last rows defined by intervals dictionary
                interval_returns[interval][level] = df.tail(days)
        self.interval_returns = interval_returns
        return interval_returns

    # Subtract risk-free rate of return from interval returns to get excess risk-free returns (Used for Sharpe/Sortino ratios) 
    def calculate_excess_rf_returns(self):
        excess_rf_returns = {}
        # Iterate through interval returns
        for interval, returns in self.interval_returns.items():
            excess_rf_returns[interval] = {}
            for level, df in returns.items():
                excess_rf_returns[interval][level] = {}
                # Subtract values in each column by risk free rate
                excess_rf_returns[interval][level] = df - self.risk_free_rate
        self.excess_rf_returns = excess_rf_returns
        return excess_rf_returns

    # Use intervals dictionary to get different samples of market (benchmark) returns
    def interval_market_returns(self):
        interval_market_returns = {}
        # Iterate through intervals, assign interval to sample of market returns
        for interval, days in self.intervals.items():
            # Get sample of most recent rows using 'days' last rows defined by intervals dictionary
            interval_market_returns[interval] = self.sp500_returns.tail(days)
        self.interval_market_returns = interval_market_returns
        return interval_market_returns
    
    # Use intervals dictionary to get different samples of factor returns
    def interval_factor_returns(self):
        interval_factor_returns = {}
        
        for interval, days in self.intervals.items():
            interval_factor_returns[interval] = self.factors

    # Subtract market returns from portfolio returns to get excess market returns (Used for Information Ratio)
    def calculate_excess_market_returns(self):
        excess_market_returns = {}
        # Match market returns with interval returns
        for interval, returns in self.interval_returns.items():
            excess_market_returns[interval] = {}
            for level, df in returns.items():
                # Ensure market_returns is aligned correctly
                market_returns = self.interval_market_returns[interval]
                # Use subtract function to apply market returns across whole df (better than iterating through columns)
                excess_market_returns[interval][level] = df.subtract(market_returns, axis=0)
                # Replace inf values, caused by subtracting market returns by null values
                excess_market_returns[interval][level] = excess_market_returns[interval][level].replace([np.inf, -np.inf], np.nan)
        self.excess_market_returns = excess_market_returns
        return excess_market_returns

    # Calculate Sharpe ratio with excess risk-free returns (currently not using any scaling factors)
    # Formula: average excess rf returns / standard deviation of excess rf returns
    def calculate_sharpe_ratio(self, trading_days_per_year = 252):
        sharpe_ratios = {}
        # Iterate through excess returns of risk-free-rate
        for interval, returns in self.excess_rf_returns.items():
            sharpe_ratios[interval] = {}

            for level, df in returns.items():
                sharpe_ratios[interval][level] = {}
                sharpe_ratios[interval][level] = (
                 #(df.mean() * trading_days_per_year) / (df.std(ddof=1) * np.sqrt(trading_days_per_year))
                 (df.mean()) / (df.std(ddof=1)) # ddof = degrees of freedom - set to 1 for sample instead of population std
                )
        self.sharpe_ratios = sharpe_ratios
        return sharpe_ratios

    # Calculate Sortino ratio with excess risk-free returns (currently not using any scaling factors)
    # Formula: average excess rf returns / standard deviation of negative excess rf returns
    def calculate_sortino_ratio(self, trading_days_per_year=252):
        sortino_ratios = {}
        # Iterate through excess returns of risk-free-rate
        for interval, returns in self.excess_rf_returns.items():
            sortino_ratios[interval] = {}

            for level, df in returns.items():
                # Find columsn with all null values and filter them out
                non_null_columns = df.columns[~df.isnull().all()]
                filtered_df = df[non_null_columns]

                sortino_ratios[interval][level] = {}

                # Calculate sortino ratios for remaining columns
                negative_excess_returns = filtered_df[filtered_df < 0]
                mean_excess_returns = filtered_df.mean() #* trading_days_per_year
                std_negative_excess_returns = negative_excess_returns.std(ddof=1) # ddof = degrees of freedom - set to 1 for sample instead of population std

                # Divide mean excess returns by std of negative excess returns
                sortino_ratios[interval][level] = mean_excess_returns / std_negative_excess_returns
                # Replace inf values with null (cause by dividing by zero)
                sortino_ratios[interval][level] = sortino_ratios[interval][level].replace([np.inf, -np.inf], np.nan)
        self.sortino_ratios = sortino_ratios
        return sortino_ratios


    # Calculate Information ratio with excess market returns (currently not using any scaling factors)
    # Formula: average excess market return / standard deviation of excess market return aka tracking error
    def calculate_information_ratio(self, trading_days_per_year = 252):
        information_ratios = {}
        # Iterate through excess returns of market
        for interval, returns in self.excess_market_returns.items():
            information_ratios[interval] = {}
            for level, df in returns.items():
                information_ratios[interval][level] = {}
                # Divide mean excess returns by std of excess market returns
                information_ratios[interval][level] = df.mean() / df.std(ddof=1) # ddof = degrees of freedom - set to 1 for sample instead of population std

        self.information_ratios = information_ratios
        return information_ratios

    # Calculate Maxmimum Drawdown (worst drop from peak) with category values (no scaling)
    def calculate_max_drawdown(self):
        max_drawdowns = {}
        for interval, days in self.intervals.items():
            max_drawdowns[interval] = {}
            for level, df in self.category_values.items():
                max_drawdowns[interval][level] = {}
                rolling_max = df.rolling(window=days, min_periods=1).max() # Finds rolling max value within days
                drawdown = (df - rolling_max) / rolling_max # Calculate drawdown at this point
                max_drawdown = drawdown.rolling(window=days, min_periods=1).min() # Returns the lowest drawdown within window
                max_drawdowns[interval][level] = max_drawdown.iloc[-1] # Store latest max drawdown value
                max_drawdowns[interval][level] = max_drawdowns[interval][level].to_dict() # Remove row name

        self.max_drawdowns = max_drawdowns
        return max_drawdowns
        
    # Calculate alpha using interval returns, betas, risk-free rate, and market returns
    # Formula for CAPM alpha: alpha = average return  - [risk-free rate of return + beta * (average benchmarket return - risk-free rate of return)]
    def calculate_alpha(self):
        alpha_values = {}
        for interval, returns in self.interval_returns.items():
            alpha_values[interval] = {}
            for level, df in returns.items():
                alpha_values[interval][level] = {}
                for category, value in df.items():
                    alpha_values[interval][level][category] = {}
                    average_market_return = self.interval_market_returns[interval].mean()
                    df = df.replace([np.inf, -np.inf], np.nan) # Replace inf values with null values
                    average_portfolio_return = df[category].mean() # Find average for each column
                    beta = self.betas[level][category] # Pull beta based on location

                    # Alpha formula: Rp - CAPM -> CAPM = rf + B * (rm - rf)
                    alpha = average_portfolio_return - (self.risk_free_rate + beta * (average_market_return - self.risk_free_rate))
                    alpha_values[interval][level][category] = alpha

        self.alpha_values = alpha_values
        return alpha_values

    # Calculate standard deviation using interval returns
    def calculate_standard_deviations(self):
        standard_deviations = {}
        for interval, returns in self.interval_returns.items():
            standard_deviations[interval] = {}
            for level, df in returns.items():
                standard_deviations[interval][level] = {}
                for category, value in df.items():
                    # ddof = degrees of freedom - set to 1 for sample instead of population std
                    standard_deviations[interval][level][category] = value.std(ddof=1) 
        self.standard_deviations = standard_deviations
        return standard_deviations

    # Calculates average return using interval returns
    def calculate_average_return(self):
        average_returns = {}
        # Iterate through interval returns to compute mean return
        for interval, returns in self.interval_returns.items():
            average_returns[interval] = {}
            for level, df in returns.items():
                average_returns[interval][level] = {}
                for category, values in df.items():
                    values = values.where(values.shift(1) != 0, np.nan).fillna(0)
                    values = values.replace([np.inf, -np.inf], np.nan)  # Replace inf values with null values
                    
                    average_returns[interval][level][category] = values.mean()

        self.average_returns = average_returns
        return average_returns

    # Calculates cumulative or total return within period using interval returns
    def calculate_cumulative_returns(self):
        cumulative_returns = {}
        for interval, returns in self.interval_returns.items():
            cumulative_returns[interval] = {}
            for level, df in returns.items():
                cumulative_returns[interval][level] = {}

                for category, series in df.items():
                    # Replace missing/infinite values before calculation
                    cleaned_series = series.replace([np.inf, -np.inf], np.nan).dropna()

                    # Calculate cumulative return: (1 + returns).cumprod() - 1
                    if not cleaned_series.empty:
                        cumulative_return = ((1 + cleaned_series).cumprod() - 1).iloc[-1] # Compounding effect in returns
                    else:
                        cumulative_return = np.nan

                    cumulative_returns[interval][level][category] = cumulative_return

        self.cumulative_returns = cumulative_returns
        return cumulative_returns
        
    # Calculate short-term beta at different time intervals
    def calculate_interval_portfolio_betas(self):
        interval_betas = {}
        for interval, returns in self.interval_returns.items(): # Iterate through each sample of daily returns
            market_returns = self.interval_market_returns.get(interval) # Get sample market returns from same interval
            interval_betas[interval] = {}
            
            for level, df in returns.items(): # Iterate through level of tables (Fund, Sector, Strategy)
                interval_betas[interval][level] = {}
                
                for category in df.columns: # Iterate through each column in each table
                    individual_returns = df[category] # Individual column of returns
                    cov = individual_returns.cov(market_returns)
                    var = market_returns.var()
                    
                    interval_betas[interval][level][category] = cov/var # beta formula = cov(portfolio returns, spy returns) / var(spy returns)
                    
        self.interval_betas = interval_betas
        return interval_betas
        
    # Calculate correlation and then square it to get r-squared estimates at different time intervals
    def calculate_interval_r_squared(self):
        interval_r_squared = {}
        for interval, returns in self.interval_returns.items(): # Iterate through each sample of daily returns
            market_returns = self.interval_market_returns.get(interval) # Get sampel market returns from same interval
            interval_r_squared[interval] = {}
            
            for level, df in returns.items(): # Iterate through level of tables (Fund, Sector, Strategy)
                interval_r_squared[interval][level] = {}
                
                for category in df.columns: # Iterate through each column in each table
                    individual_returns = df[category] # individual column of returns
                    r_squared = individual_returns.corr(market_returns) ** 2
                    interval_r_squared[interval][level][category] = r_squared
                    
        self.interval_r_squared = interval_r_squared
        return interval_r_squared

    # Calculate Treynor ratio using excess risk-free rate of return divided by beta
    def calculate_treynor_ratio(self):
        treynor_ratios = {}
        for interval, returns in self.excess_rf_returns.items(): # Iterate through each sample of excess rf returns (interval returns - risk-free rate)
            treynor_ratios[interval] = {}

            for level, df in returns.items(): # Iterate through levels of tables (Fund, Sector, Strategy)
                treynor_ratios[interval][level] = {}

                for category in df.columns:
                    excess_return = df[category].mean() # Expected or average rf return
                    beta = self.interval_betas.get(interval, {}).get(level, {}).get(category) # Get beta from self.interval_betas

                    treynor_ratio = excess_return / beta # Divide by pulled beta to get treynor ratio
                    treynor_ratios[interval][level][category] = treynor_ratio

        self.treynor_ratios = treynor_ratios
        return treynor_ratios        

# Uses to create Tracker tables with performance metrics
class PortfolioTables:
    def __init__(self, metrics):
        self.metrics_list = metrics # initialize with dictionary of PerformanceMetrics methods called in main.py script

    # Iterate through dictionarys and create dataframes with Interval, Level, Category columns + performance metrics columns
    def create_table(self, metrics_dict, metric_name):
        """Create a table for a given metric, handling both simple and category-level data."""
        tables = []
        for interval, levels in metrics_dict.items():
            for level, data in levels.items():
                # Check if the data is a dictionary (indicating category-level data) (Avg Return, Std Dev)
                if isinstance(data, dict):
                    # Flatten the dictionary (create a row for each category)
                    for category, value in data.items():
                        # Construct a DataFrame for each category
                        df = pd.DataFrame([value], columns=[metric_name])
                        df['Interval'] = interval
                        df['Level'] = level
                        df['Category'] = category
                        tables.append(df)
                else:
                    # If the data is not a dictionary (pandas Series or direct value)
                    df = pd.DataFrame(data, columns = [metric_name])  # Convert the Series to a DataFrame
                    df['Interval'] = interval
                    df['Level'] = level
                    df['Category'] = df.index
                    df.fillna(0, inplace=True)
                    tables.append(df)

        # Concatenate all the DataFrames into one
        return pd.concat(tables).reset_index(drop=True)

    # Create dictionaries to store tracker columns for each level at this point (Fund, Sector, Strategy, Portfolio, Benchmark)
    def prepare_tracker_columns(self):
        tables = {}
        # Iterate through each performance metric list item and create a column for each interval/category
        for metric_name, metric_data in self.metrics_list.items():
            tables[metric_name] = {}
            tables[metric_name] = self.create_table(metric_data, metric_name)
        return tables

    # Create tracker tables for each level
    def create_tracker_table(self, tables):
        summary_table = pd.DataFrame()
        # Combine metrics columns into one tracker table
        for metric, table in tables.items():
            table = table.set_index(['Level', 'Category', 'Interval']) # Rename indexes
            table = table.replace([np.inf, -np.inf], np.nan)
            summary_table = pd.concat([summary_table, table[metric]], axis=1)
        return summary_table

# Detaches categories from level pivot tables, creates total value tables, calculates aytd cumulative returns 
class PortfolioValueTables:
    def __init__(self, equity_pivot_tables, total_category_values, fund_values, cash_flows, sp500_data, aytd):
        self.equity_pivot_tables = equity_pivot_tables
        self.total_category_values = total_category_values
        # Makes using fund_values in ReturnsPreparer easier
        fund_values.set_index('Date', inplace=True)
        fund_values = {'Fund': fund_values}

        self.fund_values = fund_values
        self.cash_flows = cash_flows
        self.sp500_data = sp500_data
        self.aytd = aytd
        
        # Call ReturnsPreparer to get daily returns using Fund values and benchmark values
        self.portfolio_returns, self.sp500_returns = ReturnPreparer.prepare_returns(self.fund_values, self.sp500_data, 1)

    # Retrieve dates from cash flows (8/18/23 to today) to allow for reindexing 5-year pivot tables
    def retrieve_portfolio_index(self):
        # Use example index to reindex other pivot_tables
        index = self.cash_flows['Milner'].index # Using Milner instead of other funds was just a random choice
        index = index[:-1]
        return index

    # Create smaller samples of pivot tables for easier handling (separate individual funds from 'Fund')
    def prepare_pivot_tables(self):
        pivot_tables = {}
        for level, table in self.equity_pivot_tables.items():
          pivot_tables[level] = {}
          if level == "Fund": # Create pivot tables for each fund
            for fund, df in table.items():
              upper = fund[0] # Pivot table column indexes are stored as tuples, pull upper level
              pivot_table_df = table[upper]
              pivot_tables[upper] = pivot_table_df.loc[self.retrieve_portfolio_index()]
          else: # Create sector and strategy pivot tables (don't need separated individual category pivot tables)
            pivot_tables[level] = table.loc[self.retrieve_portfolio_index()]  # Filter for desired dates in index
            
        return pivot_tables

    # Create value totals tables to be compared with pivot tables
    # Ex: Milner Equity Value, Milner Cash, Milner Total, then Milner Fund pivot table with individual tickers
    # For Strategy and Sectors, just returns category totals to be compared with their respective pivot tables
    def prepare_value_tables(self):
        value_tables = {}
        for level, table in self.total_category_values.items():
          value_tables[level] = {}
          if level == "Benchmark": # Nothing to combine here
              continue
          if level == "Fund": # Create value tables for each fund
            for fund, df in table.items():
              df = df.loc[self.retrieve_portfolio_index()] # Filter for desired dates in index
              df = pd.concat([df, self.cash_flows[fund]], axis=1) # Combine with respective cash column
              df[f'{fund} Total'] = df.sum(axis=1) # Calculate total value (equity + cash)
              df.rename(columns={fund: f'{fund} Equity Value', 'Amount': f'{fund} Cash'}, inplace=True) # Rename for 'Fund' Equity Values and 'Fund' Cash
              value_tables[fund] = df
          else: # If Sector or Strategy, just filter the dates and store category totals
            value_tables[level] = table.loc[self.retrieve_portfolio_index()] # Filter for desired dates in index
          
        return value_tables

    # Calculates cumulative returns for individual funds, benchmark, and portfolio to use with Portfolio Update chart
    def prepare_fund_cumulative_returns(self):
        cumulative_returns = {}
        for level, df in self.portfolio_returns.items(): # Iterate through fund_value returns
            cumulative_returns[level] = {}
            df = df.tail(self.aytd) # Select last rows with aytd length
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            cumulative_returns[level] = (1 + df).cumprod() - 1 # Calculate cumulative returns

        cumulative_returns['Benchmark'] = {} # Create subset in dictionary for Benchmark
        # Store benchmark calculated cumulative returns output as dataframe to make joining with other columns easier
        cumulative_returns['Benchmark'] = pd.DataFrame((1 + self.sp500_returns.tail(self.aytd)).cumprod() - 1) 

        # Join dictionaries together on date
        cumulative_returns_df = pd.concat(cumulative_returns.values(), axis=1, join='outer')
        cumulative_returns_df = cumulative_returns_df.reset_index()
        # Adjust date to avoid timezones or hr/min/sec format
        cumulative_returns_df['Date'] = pd.to_datetime(cumulative_returns_df['Date'])
        cumulative_returns_df['Date'] = cumulative_returns_df['Date'].dt.date

        return cumulative_returns_df # Return dataframe of aytd cumulative returns
        
        
        
