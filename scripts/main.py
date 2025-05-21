import pandas as pd
from datetime import datetime
import os
import wrds
import numpy as np
import time
import asyncio
import warnings
warnings.filterwarnings("ignore")
from google.cloud import secretmanager
import tempfile
from google.auth import load_credentials_from_file

def access_secret(secret_name, credentials, project):
    client = secretmanager.SecretManagerServiceClient(credentials=credentials)
    project_id = project
    name = f"projects/{project_id}/secrets/{secret_name}/versions/1"
    response = client.access_secret_version(name=name)

    # Decode the secret as binary data and then decode it as base64 if needed
    secret_payload = response.payload.data.decode("UTF-8")
    return secret_payload

def create_pgpass_temp_file(pgpass_content):
    temp_pgpass_file = tempfile.NamedTemporaryFile(delete=False)
    
    with open(temp_pgpass_file.name, 'w') as f:
        f.write(pgpass_content)
    
    os.chmod(temp_pgpass_file.name, 0o600)
    return temp_pgpass_file.name

# Config path
config_path = "/app/config/config.json"
credentials, project = load_credentials_from_file(config_path)

# Retrieve the secret
google_sheets_link = access_secret("google_sheets_link", credentials, project)
folder_id = access_secret("folder_id", credentials, project)
pgpass_content = access_secret("pgpass", credentials, project)

pgpass_path = create_pgpass_temp_file(pgpass_content)
os.environ['PGPASSFILE'] = pgpass_path



from gdrivetest import run_cleaning_and_prep
from historical_data_processing import TransactionsProcessing, CashCalculations, HistoricalDataProcessing, TotalsAndWeights, PerformanceMetricsSetup, CreatePivotTables, update_transactions_df
from performance_metrics import Betas, PerformanceMetrics, PortfolioTables, PortfolioValueTables
from google_sheets_export import GoogleSheetsExporter


def main():
    """ Run classes from googledrivetest.py script """
    print('Processing gdrivetest script...')

    # Process_data class: run classes to import csvs, clean fund dfs, and turn into consolidated transactions
    consolidated_transactions_df = asyncio.run(run_cleaning_and_prep(config_path, folder_id))
    print(consolidated_transactions_df.info())
    print('run_cleaning_and_prep classes processed.')
    print('gdrivetest script finished.')

    """ Run classes from historical_data_processing.py script """
    print('Processing historical_data_processing script...')

    # TransactionsProcessing class: prepare buy/sell & grouped transactions
    processor = TransactionsProcessing(consolidated_transactions_df)
    buy_and_sell_transactions_df = processor.create_buy_and_sell_transactions_df() # Calculate running shares, start/end dates, filter for buy/sell actions
    grouped_transactions_df = processor.create_grouped_transactions_df() # Group by fund and symbol
    current_transactions_df = processor.create_current_transactions_df() # Filter holdings with shares counts above zero
    print('TransactionsProcessing class processed.')

    # CashCalculations class: prepare cash amounts over time
    cash = CashCalculations(consolidated_transactions_df)
    cash.create_total_misc_fees_df() # Aggregate fee transactions rows
    cash.create_total_fees_df() # Combine with values from fees & comm
    cash.create_fund_total_fees() # Combine misc and fees & comm
    cash.create_cash_flows() # Calculate cash flows based on net amounts by date, subtract fees

    milner_starting_cash = 310723.24
    esg_starting_cash = 2775.47
    davidson_starting_cash = 37770.43
    school_starting_cash = 627478.53
    cash_flows = cash.create_total_cash_flows(milner_starting_cash, esg_starting_cash, davidson_starting_cash, school_starting_cash) # Calculate cumulative sum using starting cash amounts
    print('CashCalculations class processed.')


    db = wrds.Connection(wrds_username='kylerzarate', autoconnect=True) # Connect to WRDS for historical prices

    # HistoricalDataProcessing class: pull prices from WRDS and create historical prices df
    historical_data_processor = HistoricalDataProcessing(grouped_transactions_df, db)
    price_start_date = historical_data_processor.get_start_date()
    historical_prices = historical_data_processor.prepare_historical_prices() # query for historical prices for each holding, clean and prepare
    market_data = historical_data_processor.get_market_data(108132, price_start_date, datetime.today()) # Enter gvkeys found on WRDs, Ex: SPY gvkey (WRDS id) == 108132
    # ijr_data = historical_data_processor.get_market_data(136584, price_start_date, datetime.today())
    # vlue_data = historical_data_processor.get_market_data(017779, price_start_date, datetime.today())
    #qual_data = historical_data_processor.get_market_data(018373, price_start_date, datetime.today())
    # mmtm_data = historical_data_processor.get_market_data(027570, price_start_date, datetime.today())
    db.close() # Close WRDS connection
    print('HistoricalDataProcessing class processed.')

    # update_transactions_df method to add WRDS Sector values to transactions dfs
    buy_and_sell_transactions_df = update_transactions_df(buy_and_sell_transactions_df, historical_prices, aggs=['Fund', "Strategy"])
    grouped_transactions_df = update_transactions_df(grouped_transactions_df, historical_prices, aggs=['Fund', "Strategy"])
    current_transactions_df = update_transactions_df(current_transactions_df, historical_prices, aggs=['Fund', "Strategy"])

    # CreatePivotTables class: prepare equity value pivot tables for Fund/Sector/Strategy levels, update summary tables with WRDS sectors
    pivot_tables = CreatePivotTables(buy_and_sell_transactions_df, historical_prices)
    historical_pivot_tables = pivot_tables.create_equity_pivot_tables() # Create equity value pivot tables by multiplying historical prices and running share counts
    equity_pivot_tables = historical_pivot_tables[0]
    price_pivot_tables = historical_pivot_tables[1]
    print('CreatePivotTables class processed.')

    # TotalsAndWeights class
    totals_and_weights = TotalsAndWeights(equity_pivot_tables, cash_flows)
    total_category_values = totals_and_weights.calculate_category_equity_values()
    fund_values = totals_and_weights.total_fund_values_aum()
    fund_values = fund_values['Fund'].loc[fund_values['Fund'].index > '8-23-2023']
    fund_values['Date'] = fund_values.index
    fund_values = fund_values.set_index('Date').reset_index()
    totals_and_weights.add_cash_columns() # Figure out where to put this when calculating weights correctly
    individual_weights = totals_and_weights.calculate_individual_weights()
    category_weights = totals_and_weights.calculate_category_weights()
    weights_within_categories = totals_and_weights.calculate_weights_within_categories()
    print('TotalsAndWeights class processed.')

    # PerformanceMetricsSetups class
    aytd = '2024-08-19'
    t_bill_code = 'TB3MS'

    metrics_setup = PerformanceMetricsSetup(aytd, t_bill_code)
    metrics_setup.calculate_calendar_aytd_length()
    metrics_setup.calculate_trading_aytd_length()
    metrics_setup.get_risk_free_rate()
    print('PerformanceMetricsSetups class processed.')
    print('historical_data_processing script finished.')

    """ Run classes from performance_metrics.py script """
    print('Processing performance_metrics script...')

    # Betas class: Calculate betas for different levels (fund, sector, strategy) using price pivot tables
    betas = Betas(price_pivot_tables, market_data, individual_weights, weights_within_categories, category_weights)
    betas.calculate_beta() # Calculate beta using covariance(portfolio, market) / variance(market) for each holding while filtering returns for trading days only
    betas.calculate_individual_weighted_betas()
    betas.calculate_category_total_weighted_betas() # Multiply each beta by corresponding weights
    betas.calculate_portfolio_total_weighted_beta() # Sum by category to get weighted betas
    print('Betas class processed.')

    # PerformanceMetrics class: Sample various intervals of returns, calculate excess returns, then use to find various performance metrics at various intervals
    metrics = PerformanceMetrics(total_category_values, betas.total_weighted_betas, market_data, metrics_setup.trading_aytd_length, metrics_setup.risk_free_rate)
    metrics.interval_returns() # Create samples of n trading days using most recent portfolio returns (Weekly: 5, Monthly: 21, AYTD: trading_aytd_length, Yearly: 252)
    metrics.calculate_excess_rf_returns() # Excess returns by subtracting scaled risk-free-rate (For Sharpe and Sortino)
    metrics.interval_market_returns() # Create samples of SPY returns to match with portfolio interval returns
    metrics.calculate_excess_market_returns() # Excess returns by subtracting interval market returns (For Info. Ratio)
    metrics.calculate_sharpe_ratio() # Calculate sharpe ratios, annualize average excess returns divided by std of excess returns
    metrics.calculate_sortino_ratio() # Calculate sortino ratios, annualize average excess returns divided by std of negative excess returns
    metrics.calculate_information_ratio() # Calculate info. ratio, average excess market returns dividng by std of excess market returns
    metrics.calculate_max_drawdown() # Find MDD for intervals, find minimum return within interval samples
    metrics.calculate_alpha() # Find alpha for intervals, find by subtracting average return by expected return (CAPM)
    metrics.calculate_standard_deviations() # Find standard deviation for intervals
    metrics.calculate_average_return() # Find average return for intervals
    metrics.calculate_cumulative_returns() # Find cumulative return for each interval
    metrics.calculate_interval_portfolio_betas() # Calculate short-term betas at different intervals of time
    metrics.calculate_interval_r_squared() # Calculate r2 from correlations
    metrics.calculate_treynor_ratio() # Calculate Treynor Ratios using excess rf returns and beta
    print('PerformanceMetrics class processed.')

    performance_metrics_list = {
	'Returns': metrics.average_returns,
        'Cumulative Returns': metrics.cumulative_returns,       
        'Alpha': metrics.alpha_values,
        'Beta': metrics.interval_betas,
        'Sharpe Ratio': metrics.sharpe_ratios,
        'Standard Deviation': metrics.standard_deviations,
        'MDD' : metrics.max_drawdowns,
        'Information Ratio': metrics.information_ratios,
        'Sortino Ratio': metrics.sortino_ratios,
        'Treynor Ratio': metrics.treynor_ratios,
        'R-Squared': metrics.interval_r_squared
    }

    # PortfolioTables class: Prepare summary tables to be used in Google Sheets file
    portfolio_tables = PortfolioTables(performance_metrics_list)
    tracker_table = portfolio_tables.create_tracker_table(portfolio_tables.prepare_tracker_columns()) # Adds metrics columns from performance_metrics_list to dataframe
    tracker_table.reset_index(inplace=True)
    tracker_table.rename(columns={'level_0': 'Level', 'level_1': 'Category', 'level_2': 'Interval'}, inplace=True)
    print('PortfolioTables class processed.')

    # PortfolioValueTables class: Prepare value tables to be used in Google Sheets file
    value_tables = PortfolioValueTables(equity_pivot_tables, total_category_values, fund_values, cash_flows, market_data, metrics_setup.trading_aytd_length)
    pivot_tables_dict = value_tables.prepare_pivot_tables()
    value_tables_dict = value_tables.prepare_value_tables()
    cumulative_returns_df = value_tables.prepare_fund_cumulative_returns()
    print('PortfolioValueTables class processed.')
    print('performance_metrics script finished.')

    """ Run classes from google_sheets_export.py script """
    print('Processing google_sheets_export script...')

    # GoogleSheetsExporter class: pull credentials, access spreadsheet, and export relevant tables to spreadsheet
    exporter = GoogleSheetsExporter(google_sheets_link, config_path)
    
    # Define dataframes to export with export_summary_tables_to_google_sheets() method from GoogleSheetsExporter
    summary_tables_to_export = {
        'Summary Holdings': current_transactions_df['Fund'],
        'Consolidated Transactions': consolidated_transactions_df.sort_values(by='Date', ascending=False),
        'S&P Data': market_data.reset_index(),
        'AYTD Cumulative Returns': cumulative_returns_df,
        'Portfolio Tracker': tracker_table[tracker_table['Level']=='Portfolio'],
        'Benchmark Tracker': tracker_table[tracker_table['Level']=='Benchmark'],
        'Fund Tracker': tracker_table[tracker_table['Level']=='Fund'],
        'Sector Tracker': tracker_table[tracker_table['Level']=='Sector'],
        'Strategy Tracker': tracker_table[tracker_table['Level']=='Strategy'],
	    'Portfolio Values': fund_values.reset_index(),
	    'Individual Weights': pd.DataFrame(individual_weights['Fund'].iloc[[-1]].fillna(0).T).reset_index().rename(columns={'index': 'Ticker', individual_weights['Fund'].index[-1]: 'Weight'}).sort_values(by='Weight', ascending=False)

    }
    
    # Define tuples of dataframes (df1, df2) to export with export_value_tables_to_google_sheets() method from GoogleSheetsExporter
    value_tables_to_export = {
        # Transpose, reset index, and rename beta tables
        'Fund Betas' : (
                    pd.DataFrame(betas.total_weighted_betas['Fund'], index=[0])
                        .T.reset_index()
                        .rename(columns={'index': 'Fund', 0: 'Beta'}),
                    pd.DataFrame(betas.betas['Fund'], index=[0])
                        .T.reset_index()
                        .rename(columns={'level_0': 'Fund', 'level_1': 'Symbol', 0: 'Beta'})
                ),
        'Sector Betas' : (
                    pd.DataFrame(betas.total_weighted_betas['Sector'], index=[0])
                        .T.reset_index()
                        .rename(columns={'index': 'Fund', 0: 'Sector'}),
                    pd.DataFrame(betas.betas['Sector'], index=[0])
                        .T.reset_index()
                        .rename(columns={'level_0': 'Fund', 'level_1': 'Symbol', 0: 'Beta'})
                ),
        'Strategy Betas' : (
                    pd.DataFrame(betas.total_weighted_betas['Strategy'], index=[0])
                        .T.reset_index()
                        .rename(columns={'index': 'Fund', 0: 'Beta'}),
                    pd.DataFrame(betas.betas['Strategy'], index=[0])
                        .T.reset_index()
                        .rename(columns={'level_0': 'Fund', 'level_1': 'Symbol', 0: 'Beta'})
                ),
        # Create tables for Start and End Dates by grouping by first Start Date and last End Date from grouped_transactions_df
        'Start Dates': (
                    pd.DataFrame(grouped_transactions_df['Fund'].groupby(['Sector']).agg({'Start Date': 'first', 'End Date': 'last'}).sort_values(by=['End Date', 'Start Date'], ascending=False)).reset_index(),
                    pd.DataFrame(grouped_transactions_df['Strategy'].groupby(['Strategy']).agg({'Start Date': 'first', 'End Date': 'last'}).sort_values(by=['End Date', 'Start Date'], ascending=False)).reset_index()
                ),
        # Value total df and pivot table combos
        'Strategy' : (value_tables_dict['Strategy'].reset_index(), pivot_tables_dict['Strategy'].reset_index()),
        'Sector' : (value_tables_dict['Sector'].reset_index(), pivot_tables_dict['Sector'].reset_index()),
        'Milner' : (value_tables_dict['Milner'].reset_index(), pivot_tables_dict['Milner'].reset_index()),
        'ESG' : (value_tables_dict['ESG'].reset_index(), pivot_tables_dict['ESG'].reset_index()),
        'Davidson' : (value_tables_dict['Davidson'].reset_index(), pivot_tables_dict['Davidson'].reset_index()),
        'School' : (value_tables_dict['School'].reset_index(), pivot_tables_dict['School'].reset_index())
    }

    weights_within_categories['Fund'] = weights_within_categories['Fund'].fillna(0)
    # Replace inf and -inf with 0 or None
    weights_within_categories['Fund'].replace([np.inf, -np.inf], 0, inplace=True)

    # Define tuples of dataframes (df1, df2) used to export latest weights with export_value_tables_to_google_sheets() method from GoogleSheetsExporter
    weight_tables_to_export = {
        # Create dataframe by retrieving last date from weights pivot table, then transpose, reset index, rename, and sort values
        'Strategy Weights': (
            pd.DataFrame(category_weights['Strategy'].iloc[-1].T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Strategy'].index[-1]: 'Weight'})
                .sort_values(by='Weight', ascending=False),
            pd.DataFrame(weights_within_categories['Strategy'].iloc[-1].T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Strategy'].index[-1]: 'Weight'})
                .sort_values(by=['Strategy','Weight'], ascending=[True, False])
        ),
        'Sector Weights': (
            pd.DataFrame(category_weights['Sector'].iloc[-1].T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Sector'].index[-1]: 'Weight'})
                .sort_values(by='Weight', ascending=False),
            pd.DataFrame(weights_within_categories['Sector'].iloc[-1].T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Sector'].index[-1]: 'Weight'})
                .sort_values(by=['Sector','Weight'], ascending=[True, False])
        ),
        'Fund Weights' : (
            pd.DataFrame(category_weights['Fund'].iloc[-1].fillna(0).T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Fund'].index[-1]: 'Weight'})
                .sort_values(by='Weight', ascending=False),
            pd.DataFrame(weights_within_categories['Fund'].iloc[-1].fillna(0).T)
                .reset_index()
                .rename(columns={'index': 'Ticker', category_weights['Fund'].index[-1]: 'Weight'})
                .sort_values(by=['Fund','Weight'], ascending=[True, False])
        )
    }

    # Iterate through items in tables dictionaries, calling GoogleSheetsExporter methods for each item
    for sheet_name, df in summary_tables_to_export.items():
        exporter.export_summary_tables_to_google_sheets(sheet_name, df)
        time.sleep(5) # Wait a couple seconds to avoid overloading Google Sheets API

    for sheet_name, df in value_tables_to_export.items():
        exporter.export_value_tables_to_google_sheets(sheet_name, df)
        time.sleep(5) # Wait a couple seconds to avoid overloading Google Sheets API

    for sheet_name, df in weight_tables_to_export.items():
        exporter.export_value_tables_to_google_sheets(sheet_name, df)
        time.sleep(5) # Wait a couple seconds to avoid overloading Google Sheets API

    print('GoogleSheetsExporter class processed.')
    print('google_sheets_export script finished.')

    print('All done.')
    return summary_tables_to_export, value_tables_to_export

if __name__ == '__main__':
  main()

