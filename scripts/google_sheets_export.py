import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
import pandas as pd
from datetime import datetime
import numpy as np
import pandas_market_calendars as mcal
import warnings
import time
warnings.filterwarnings("ignore")

# Defines different processes for writing dataframes to Google Sheets
class GoogleSheetsExporter:
    def __init__(self, sheet_name, credentials_json):
        self.sheet_name = sheet_name # Google Sheets link - stored as secret or txt files
        self.credentials_json = credentials_json # config.json
        self.client = self.authenticate_google_sheets() # Method for authenticating Google Sheets API with credentials

    # Authenticates Google Sheets API
    def authenticate_google_sheets(self):
        # Define the scope for Google Sheets and Drive API
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # Prepare credentials for client object using config.json file containing credentials/secrets
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_json, scope)
        # Authenticate Service Account (which iteracts with the Google Sheet and writes the dfs using gspread) with prepared credentials
        client = gspread.authorize(credentials)
        return client

    # Use Service Account to access Google Sheet with input link
    def open_spreadsheet(self):
        spreadsheet = self.client.open_by_url(self.sheet_name)
        return spreadsheet

    # Exports one table to one sheet (for use with individual dfs like current holdings, consolidated transactions, aytd cumulative returns, etc. - not dictionaries)
    def export_summary_tables_to_google_sheets(self, sheet_name, df):
        spreadsheet = self.open_spreadsheet() # Opens Google Sheet
        try:
            # Try to open the specific worksheet by name
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            # If the worksheet doesn't exist, create a new one with dimensions defined by rows/columns in input dataframe
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=df.shape[0], cols=df.shape[1])
        worksheet.clear() # Clear worksheet before writing so we don't get leftover values
        time.sleep(2) # Pause for a couple second so we don't overload the api
        set_with_dataframe(worksheet, df) # gspread_dataframe function for writing to Google Sheets
        
        # Confirm df was written to Google Sheets
        print(f"Data has been successfully exported to the '{sheet_name}' sheet in the Google Sheet '{self.sheet_name}'.")

    # Exports one value table and one pivot table created in PortfolioTables and PortfolioValuetables classes
    def export_value_tables_to_google_sheets(self, sheet_name, df):
        spreadsheet = self.open_spreadsheet() # Opens Google Sheet
        try:
            # Try to open the worksheet by name
            worksheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            # If the worksheet doesn't exist, create a new one
            max_rows = max(df[0].shape[0], df[1].shape[0]) # Enough rows to fit the longest dataframe
            max_cols = df[0].shape[1] + df[1].shape[1] # enough columns to fit both dataframes together plus an additional column for spacing
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows = max_rows, cols = max_cols)
        worksheet.clear() # Clear worksheet before writing so we don't get leftover values
        time.sleep(2) # Pause for a couple second so we don't overload the api
        set_with_dataframe(worksheet, df[0], row=1, col=1) # Write the first dataframe at cell A1
        set_with_dataframe(worksheet, df[1], row=1, col=df[0].shape[1] + 2) # Write second dataframe next to first df, with 1 space between
        
        # Confirm df was written to Google Sheets
        print(f"Data has been successfully exported to the '{sheet_name}' sheet in the Google Sheet '{self.sheet_name}'.")