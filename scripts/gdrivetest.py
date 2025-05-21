import aiohttp
import asyncio
import pandas as pd
from io import StringIO
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import warnings
warnings.filterwarnings("ignore")



# Parent class: Common reading processes between funds
class FundReadCSV:
    def __init__(self, url):
        self.url = url
        self.df = None

    async def load_and_process(self):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def clean_dates(df):
        for col in df.columns:
            if 'date' in col.lower(): # Some 'Date' rows have text "as of", this splits and returns only the first date
                df[col] = pd.to_datetime(df[col].astype(str).str.split(' as of ').str[0], errors='coerce').dt.strftime('%Y-%m-%d')
        return df


# Milner fund specific csv reading child class
class MilnerReadCSV(FundReadCSV):
    async def load_and_process(self):
        # If DataFrame is already provided, just clean dates
        if self.df is not None:
            return self.clean_dates(self.df)

        # Fetch csv files asynchronously from urls
        async with aiohttp.ClientSession() as session: # This starts session to read urls
            async with session.get(self.url) as response: # Sends a GET request to urls
                csv_data = await response.text() # This reads the url contents as text
                df = pd.read_csv(StringIO(csv_data), index_col=None, thousands=',', dtype=str) # Loads text into a dataframe, use dtype=str for faster processing
                return self.clean_dates(df) # Clean dates and return the processed dataframe

# ESG fund specific csv reading child class
class ESGReadCSV(FundReadCSV):
    async def load_and_process(self):
        # If DataFrame is already provided, just clean dates
        if self.df is not None:
            return self.clean_dates(self.df)
 
        # Fetch csv files asynchronously from urls
        async with aiohttp.ClientSession() as session: # same process as MilnerReadCSV
            async with session.get(self.url) as response:
                csv_data = await response.text()

                # Read first line to check for 'Account'
                first_line = csv_data.split('\n')[0].strip() 
                
                # This is to adjust for the trading account change altering csv formats
                skip_rows = 3 if 'Account' in first_line else 0 # Decides to skip rows to avoid "Account" cell in csvs if present
                
                df = pd.read_csv(StringIO(csv_data), skiprows=skip_rows, index_col=None, thousands=',', dtype=str)
                df.dropna(how='all', inplace=True) # dropna for cleaner data
                
                return self.clean_dates(df)

# Davidson fund specific csv reading child class
class DavidsonReadCSV(FundReadCSV):
    async def load_and_process(self):
        # If DataFrame is already provided, just clean dates
        if self.df is not None:
            return self.clean_dates(self.df)

        # Fetch csv files asynchronously from urls 
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                csv_data = await response.text()
                df = pd.read_csv(StringIO(csv_data), usecols=range(11), thousands=',', dtype=str) # Use first 11 columns
                return self.clean_dates(df)

# School fund specific csv reading child class
class SchoolReadCSV(FundReadCSV):
    # If DataFrame is already provided, just clean dates
    async def load_and_process(self):
        if self.df is not None:
            return self.clean_dates(self.df)
            
        # Fetch csv files asynchronously from urls
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                csv_data = await response.text()

                # Read first line to check for 'Account'
                first_line = csv_data.split('\n')[0].strip()
                
                # This is to adjust for the trading account change altering csv formats
                skip_rows = 3 if 'Account' in first_line else 0 # Decides to skip rows to avoid "Account" cell in csvs if present
                
                df = pd.read_csv(StringIO(csv_data), skiprows=skip_rows, index_col=None, thousands=',', dtype=str)
                df.dropna(how='all', inplace=True) # dropna for cleaner data
                
                return self.clean_dates(df)

# Parent class: Common cleaning processes between funds
class ColumnCleaner:
    def __init__(self, df):
        self.df = df
        self.clean_columns() # Immediately apply the cleaning steps

    def clean_columns(self):
        self.df.drop_duplicates(keep='first', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Dictionary for standardizing column names
        rename_dict = {
            'TransactionDate': 'Date',
            'Transaction Date': 'Date',
            'Commission': 'Fees & Comm',
            'Commission/Fees': 'Fees & Comm',
            'TransactionType': 'Action',
            'Type': 'Action',
            'QTY': 'Quantity'
        }
        # Rename column if original column name is found as a dictionary key
        self.df.rename(columns={col: rename_dict[col] for col in self.df.columns if col in rename_dict}, inplace=True)
        self.df = self.df.groupby(level=0, axis=1).first() # Combine duplicate columns with the same name if they exist
        if 'Date' in self.df.columns: # If 'Date' column exists (sometimes the timing with renaming is off), convert to a standardized short date format
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Convert specified columns to numeric columns after removing dollar signs and commas
        cols_to_numeric = ['Price', 'Quantity', 'Fees & Comm', 'Amount']
        self.df[cols_to_numeric] = self.df[cols_to_numeric].replace('[\$,]', '', regex=True).apply(pd.to_numeric, errors='coerce')

        # 
        if 'Strategy' in self.df.columns: # If 'Strategy' exists, replace 'Old' with 'Fundamental Analysis', then forward fill null strategies within each Symbol with first defined strategy
            self.df.loc[self.df['Strategy'] == 'Old', 'Strategy'] = 'Fundamental Analysis'
            self.df['Strategy'] = self.df['Strategy'].fillna(self.df.groupby('Symbol')['Strategy'].transform('first'))

        column_order = ['Date', 'Action', 'Symbol', 'Quantity', 'Price', 'Fees & Comm', 'Amount', 'Strategy', 'Sector', 'Fund', 'Description']
        self.df = self.df[[col for col in column_order if col in self.df.columns]] # Reorder columnb based on specified column order
        self.df.sort_values(by=['Date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

# Milner fund specific column cleaning child class
class MilnerColumnCleaner(ColumnCleaner):
    def __init__(self, df):
        super().__init__(df)
        if 'Symbol' in self.df.columns:
            # If Action is 'Buy' and strategy is null, then fill with 'Fundamental Analysis'
            self.df.loc[(self.df['Action'] == 'Buy') & (self.df['Strategy'].isna()) & (self.df['Symbol'] != '78486Q101'), 'Strategy'] = 'Fundamental Analysis'
        self.df.drop(columns=['Description'], errors='ignore', inplace=True)
        
# ESG fund specific column cleaning child class
class ESGColumnCleaner(ColumnCleaner):
    def __init__(self, df):
        super().__init__(df)
        if 'Description' in self.df.columns:
            self.df['Description'].fillna('', inplace=True) # Replace missing descriptions with empty strings (need this for LRCX fix)
            self.df.loc[self.df['Description'].str.contains('LAM RESEARCH CORP'), 'Symbol'] = 'LRCX' # Fix specifc LRCX symbol (Weird formatting)
            self.df.drop(columns=['Description'], errors='ignore', inplace=True) # Drop 'Description' column
        if 'Strategy' in  self.df.columns:
            # If Action is 'Buy' and strategy is null, then fill with 'Fundamental Analysis'
            self.df.loc[(self.df['Strategy'].isnull()) & self.df['Action'].isin(['Bought', 'Sell']), 'Strategy'] = 'Fundamental Analysis'
            # Forward fill null strategies within each Symbol with first defined strategy
            self.df['Strategy'] = self.df['Strategy'].fillna(self.df.groupby('Symbol')['Strategy'].transform('first'))

# Davidson fund specific column cleaning child class
class DavidsonColumnCleaner(ColumnCleaner):
    def __init__(self, df):
        super().__init__(df)
        if 'Sector' in self.df.columns: # Other funds have a 'Sector' column, this empty column just makes joining easier later
            self.df[['Sector']] = None
        self.df.drop(columns=['Cash/Margin', 'Account', 'SettleDate', 'Description'], errors='ignore', inplace=True) # Drop useless columns

        if 'Fees & Comm' in self.df.columns:
            self.df['Fees & Comm'] = self.df['Fees & Comm'] / abs(self.df['Quantity']) # 'Fees & Comm' represents cost per share
            self.df['Fees & Comm'] = self.df['Fees & Comm'].fillna(0)

# School fund specific column cleaning child class
class SchoolColumnCleaner(ColumnCleaner):
    def __init__(self, df):
        super().__init__(df)
        self.df.drop(columns=['SecurityType', 'Description'], errors='ignore', inplace=True) # Drop useless columns
        self.df.dropna(subset=['Date'], inplace=True) # Remove rows with empty dates

# Base class to handle CSV processing
class FundDataProcessor:
    def __init__(self, max_concurrent_requests=10):
        # Map fund names to their respective reader classes
        self.fund_read_classes = {
            'Milner': MilnerReadCSV,
            'ESG': ESGReadCSV,
            'Davidson': DavidsonReadCSV,
            'School': SchoolReadCSV
        }
        # Map fund names to their respective column cleaner classes
        self.column_clean_classes = {
            'Milner': MilnerColumnCleaner,
            'ESG': ESGColumnCleaner,
            'Davidson': DavidsonColumnCleaner,
            'School': SchoolColumnCleaner
        }
        # Semaphore to limit the number of concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def process_single_fund_csv(self, url, fund_read_class):
        # Process a single CSV file with concurrency control
        async with self.semaphore:
            # Create a FundReadCSV object and load/process the csv data asynchronously
            fund_reader = fund_read_class(url=url)
            return await fund_reader.load_and_process()

    async def process_fund_csvs(self, fund_name, urls):
        # Apply correct reader and column classes based on fund name
        fund_read_class = self.fund_read_classes.get(fund_name)
        column_clean_class = self.column_clean_classes.get(fund_name)

        # Create async tasks to fetch and process csvs for all funds, apply fund reader
        tasks = [self.process_single_fund_csv(url, fund_read_class) for url in urls]
        fund_dataframes = await asyncio.gather(*tasks) # Wait for all tasks to complete

        combined_df = pd.concat(fund_dataframes, ignore_index=True) # Concatenate all DataFrames into one
        combined_df['Fund'] = fund_name # Create 'Fund' column to identify funds
        return column_clean_class(combined_df).df # Apply column cleaner

    async def download_and_process_csvs(self, import_urls):
        # Create async tasks to download and process all funds from generated urls
        tasks = [self.process_fund_csvs(fund_name, list(urls.values())) for fund_name, urls in import_urls.items()]
        results = await asyncio.gather(*tasks) # Wait for all funds to finish processing
        # Return dictionary of fund names to respective cleaned dataframes
        return {fund_name: df for fund_name, df in zip(import_urls.keys(), results)}

# Class to authenticate and retrieve file metadata from Google Drive folder
class GoogleDriveFetcher:
    def __init__(self, credentials_file, folder_id):
        # Initialize with service account credentials from Google Cloud and target folder id (Google Drive folder for Brogaard's Emails)
        self.credentials_file = credentials_file
        self.folder_id = folder_id
        self.service = self._authenticate_drive() # Authenticate with Google Drive API

    def _authenticate_drive(self): 
        # Authenticate to Google Drive using service account credentials
        scope = ["https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scope) # Read from config.json file
        return build('drive', 'v3', credentials=credentials)

    def fetch_csv_files_from_folder(self):
        csv_files = [] # List to store retrieved CSV files

        def fetch_folder_contents(folder_id):
            # Recursively fetch all files/folders inside the target folder id
            query = f"'{folder_id}' in parents and trashed = false" # Query for searching for present files in Google Drive folder
            # Execute query, returns metadata for each file (file id, file name, file type)
            results = self.service.files().list(q=query, pageSize=1000, fields="files(id, name, mimeType)").execute()
            files = results.get('files', [])
            
            # Iterate through list to identify folders and csv files
            for file in files:
                if file['mimeType'] == 'application/vnd.google-apps.folder': # Identifies folder
                    print(file['name']) # Print folder name to make sure all folders are being read
                    fetch_folder_contents(file['id']) # Repeat method to drill into folder
                elif file['mimeType'] == 'text/csv':
                    csv_files.append(file) # If file is a csv, add it to csv_files list

        fetch_folder_contents(self.folder_id) # Start fetching process from initial target folder
        return csv_files


# Class to generate URLs for CSV files in Google Drive based on metadata
class CSVUrlGenerator:
    def __init__(self, files):
        self.files = files # Initialize with list of Google Drive file metadata
        
    def generate_urls(self, funds):
        # Stores generated urls
        import_urls = {}
        for fund in funds: 
            import_urls[fund] = {} # Add level of dictionaries for each fund
            for file in self.files:
                if str(fund).lower() in str(file['name']).lower(): # If Fund name in file name: 
                    # Create string for url by joining generic download format and file id for the csv file
                    url = f'https://drive.google.com/uc?export=download&id={file["id"]}' 
                    import_urls[fund][file['name']] = url
        return import_urls # Return dictionaries of funds containing respective urls

# Class to consolidate transaction datasets form each fund into one dataframe
class ConsolidatedTransactions:
    def __init__(self, datasets):
        self.datasets = datasets # Dictionary of fund dataframes
        self.consolidated_transactions_df = None

    def create_consolidated_transactions(self):
        # Combine datasets into one dataframe
        consolidated_transactions_df = pd.concat(self.datasets, ignore_index=True)
        # Convert all 'Buy' values in Action column to 'Bought
        consolidated_transactions_df['Action'] = consolidated_transactions_df['Action'].replace({'Buy': 'Bought','Sold': 'Sell'})

        # Convert all 'Share BuyBack' values in Strategy to 'Share Buybacks'
        consolidated_transactions_df['Strategy'] = consolidated_transactions_df['Strategy'].replace({'Share Buyback': 'Share Buybacks','Share BuyBack': 'Share Buybacks'})

        # Make every Quantity value negative when Action is sell, assuming some negatives are present
        consolidated_transactions_df.loc[consolidated_transactions_df['Action'] == 'Sell', 'Quantity'] = abs(consolidated_transactions_df['Quantity']) * -1

        # Make every Amount value positive when Action is sell, assuming some negatives are present
        consolidated_transactions_df.loc[consolidated_transactions_df['Action'] == 'Sell', 'Amount'] = abs(consolidated_transactions_df['Amount'])

        # Round prices to dollar amounts, 2 places
        consolidated_transactions_df['Price'] = consolidated_transactions_df['Price'].round(2)

        # Filter out 'Security Transfer' columns that mess with cash amounts
        consolidated_transactions_df = consolidated_transactions_df[consolidated_transactions_df['Action'] != 'Security Transfer']

        # Handle different formats for date entry 'm/dd/yy' vs 'm/dd/yyyy'
        consolidated_transactions_df['Date'] = pd.to_datetime(consolidated_transactions_df['Date'], format='mixed')

        # Remove duplicate rows
        consolidated_transactions_df = consolidated_transactions_df.drop_duplicates(subset=['Date', 'Action', 'Symbol', 'Quantity', 'Fund'], keep='last')
        # Remove 'Adjustment' actions
        consolidated_transactions_df = consolidated_transactions_df[consolidated_transactions_df['Action'] != 'Adjustment']
        consolidated_transactions_df.drop(columns=['Sector'], inplace=True) # Drop Sector column (useless because we'll pull sectors from WRDS)
        self.consolidated_transactions_df = consolidated_transactions_df

        return consolidated_transactions_df # Return one dataframe with all the cleaned fund transactions

# Main function to orchestrate execution of fetching, cleaning, and preparing the consolidated transactions
async def run_cleaning_and_prep(credentials_file, folder_id):
    # Step 1: Fetch CSV files from Google Drive
    drive_fetcher = GoogleDriveFetcher(credentials_file, folder_id) # Call and set up GoogleDriveFetcher class
    csv_files = drive_fetcher.fetch_csv_files_from_folder() # Create list of csv file metadata using fetch_csv_files_from_folder method

    # Step 2: Generate download URLs for all fetched csv files
    url_generator = CSVUrlGenerator(csv_files) # Call and set up CSVUrlGenerator class
    import_urls = url_generator.generate_urls(funds=['Milner', 'ESG', 'Davidson', 'School']) # Create dictionary of urls using generate_urls method to define each fund manually

    # Step 3: Download and process csv data concurrently
    fund_processor = FundDataProcessor(max_concurrent_requests=10) # Call and set up FundDataProcessor class with defined concurrent tasks limit (10 is currently ok)
    datasets = await fund_processor.download_and_process_csvs(import_urls) # Create dictionary of fund datasets using download_and_process_csvs method
    
    # Step 4: Consolidate transactions across all funds
    consolidated_transactions = ConsolidatedTransactions(datasets) # Call and set up ConsolidatedTransactions class with fund datasets
    consolidated_transactions_df = consolidated_transactions.create_consolidated_transactions() # Create consolidated transactions dataframe using create_consolidated_transactions method
    
    # Step 5: return final consolidated dataframe
    return consolidated_transactions_df




