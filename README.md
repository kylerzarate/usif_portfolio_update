# U of U Student Investmend Fund Portfolio Update Project
This project was created to handle transactions stored in csv files, clean and prepare data for different funds, and calculate various weights and performance metrics at different intervals and aggregations of data. Historical prices and sector information are pulled from the Wharton Research Data Services database: https://wrds-www.wharton.upenn.edu/

**Link to project:** 

## How It's Made:

**Tech used:** Python, SQL, Google Sheets
Portfolio Update Documentation

## General Overview:

This set of scripts (gdrivetest, historical_data_processing, performance_metrics, google_sheet_export) each handle a different stage of the overall process. This overall process is orchestrated by main.py, where modules from each script are imported and called throughout. Each script uses classes to create an outline for various processes. Within each class, there are functions that can reference each other. At the end of the school year, the scripts were deployed to a Google Cloud Run Job.

**“gdrivetest”** is used to access Google Drive without having to mount the drive in Google Colab, making it easier to share with others. Once accessed, the service account created on Google Cloud is used to pull file ids of csv files. Instead of downloading the whole file, the files are read using website links generated by a concatenation of a Google Drive URL template and the file id of the csv files. These csv files are then cleaned and prepared to a common format into what’s called “consolidated_transactions_df”. This consolidated transaction data is then used as a foundation for the other scripts. 

**“historical_data_processing”** manipulates the consolidated transactions into tables used to perform other calculations, such as grouping to get start/end dates and a running count of shares and filtering for transactions to calculate cash flow over time. This prepared data is then used to build a SQL query used to access the Wharton Research Data Services (WRDS) Compustat database. Three following queries are pulled: 
1) available historical prices based on defined start and end dates for each ticker to be used in building portfolio values, 
2) 5 years of historical prices based on available start dates and defined end dates to be used to calculate betas, and
3) 5 years of historical SPY values to help with calculating metrics
After filtering for relevant securities, the historical price data is used to create historical price, shares, and equity value pivot tables, with dates as rows and tickers as columns. This is done through each “level”, which refers to different aggregations of the data. “Fund” refers to the total equity values of each fund Milner, ESG, School, and Davidson. This is also done through “Sector” and “Strategy”, where the pivot tables use Sector and Strategy columns instead of the Fund column. From here, the totals and proportions for each level and category are calculated and relevant data is pulled to prepare for the next script. 

**“performance_metrics”** takes the various tables built so far and uses them to calculate statistics for the fund. To start, a returns preparer method is used to create samples of returns throughout the different levels. This method can be iterated through different samples of weekly, monthly, academic-year-to-date, and yearly returns. Using the output of five year individual tickers and S&P 500 prices, monthly returns can be used to calculate a beta in a method similar to Yahoo Finance (Five Year Monthly Betas).


**“Google_sheets_export”** uses the Google Sheets API and service account credentials to access the defined Google Sheet and export Dataframes to defined sheets. There are two types of exports, “summary_tables” and “value_tables”, which differ depending on the amount of tables exported to a sheet. Summary tables are reserved for sheets with only one table per sheet, such as consolidated transactions, summary holdings, metrics trackers, and others. Value tables are reserved sheets that use multiple tables, such as the fund-specific, sector, and strategy sheets, where a consolidated total value table and a pivot table with individual holdings are exported to the same sheet to allow users to crossreference.


## Optimizations
**Implemented improvements:**

1. Originally, this project was written in one long Google Colab script. With unorganized code, it was necessary to refactor into five smaller scripts to improve readability as well as rethink the data processing.
2. Classes were used to allow for inheritance and handling different levels of aggregated data, such as calculating metrics for holdings within strategies and sectors which was not available before in the fund.
3. Google Cloud was used to schedule automatic executions of the scripts and track logs, freeing up time that was spent manually starting and verifying the executions. Also reduced the costs of using cloud services to just several cents a month by being selective about resources used.
4. Originally, this project tried to use Yahoo Finance api for historical prices. Given the large number of holdings in the fund, the api would be overloaded at times and make the execution run slow. After some research, WRDS database was implemented, creating SQL calls embedded within the scripts to pull large amounts of data.

**Potential Improvements:**

Problem Areas (things to improve the functionality):
1. Get away from hard-coded “start date” of 8/18/2023, also will allow us to do better historical analysis. This was more of a data availability problem than a programming problem, as we decided on a hard reset in cash and time of holdings at that time.
2. Set up more interactive dashboard with Power BI, moving away from using Google Sheets.
3. Calculate performance metrics over time to get rolling stats. These are currently just a point in time analysis (last week/month/year) - can we see how beta changes over time?

## Lessons Learned:

1. **Ensuring data quality is just as important as the program's capabilities:** Many times, a problem in the output was because of a data issue than a problem with the program itself. I found that identifying and fixing problems closer to the source was a much better call compared to duct-tape solutions at the end, even if it was harder to figure out.
2. **Implementing object-oriented programming:** Refactoring the code to use classes was more of a technical challenge, but it paid off when parts of the processing needed to be adjusted. It was much easier to identify the places that needed to be adjusted when the code was organized.
3. **Figuring out how to use Docker/Google Cloud:** What started out as a way to call the Google Sheets api eventually turned into exploring more of Google Clouds services and deploying the script in a Google Cloud Run Job. I figured out how images and containerization works in Docker, how to create an image, add a tag, and then push it to a repository in Google Cloud.
