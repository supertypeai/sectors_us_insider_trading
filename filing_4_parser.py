import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
from dotenv import load_dotenv
from edgar import *
from neon_connector.neon_connector import NeonConnector
from datetime import date, timedelta


def parse_form_4_filing(accession_number, xml_content):
    pd.options.mode.copy_on_write = True
    
    def get_identity(text):
        infoList = list()
        
        issuer = text.find("issuer")
        if issuer.find("issuerCik") is not None:
            infoList.append(["issuerCik", issuer.find("issuerCik").text.strip()])
        
        owner = text.find("reportingOwnerId")
        if owner.find("rptOwnerName") is not None:
            infoList.append(["executiveName", owner.find("rptOwnerName").text.strip()])
        if owner.find("rptOwnerCik") is not None:
            infoList.append(["rptOwnerCik", owner.find("rptOwnerCik").text.strip()])
        
        relationship = text.find("reportingOwnerRelationship")
        if relationship.find("isDirector") is not None:
            infoList.append(["isDirector", relationship.find("isDirector").text.strip()])
        if relationship.find("isOfficer") is not None:
            infoList.append(["isOfficer", relationship.find("isOfficer").text.strip()])
        if relationship.find("isTenPercentOwner") is not None:
            infoList.append(["isTenPercentOwner", relationship.find("isTenPercentOwner").text.strip()])
        if relationship.find("isOther") is not None:
            infoList.append(["isOther", relationship.find("isOther").text.strip()])
        if relationship.find("officerTitle") is not None:
            infoList.append(["officerTitle", relationship.find("officerTitle").text.strip()])

        if text.find("periodOfReport") is not None:
            infoList.append(["periodOfReport", text.find("periodOfReport").text.strip()])
        if text.find("remarks") is not None:
            infoList.append(["remarks", text.find("remarks").text.strip()])
            
        dataDict = dict()
        for item in infoList:
            dataDict[item[0]] = item[1]
            
        data = pd.DataFrame.from_dict([dataDict])      
        return(data)
    
    def get_transaction_row(transaction):
        infoTransaction = list()
        
        if transaction.find("securityTitle") is not None:
            infoTransaction.append(["securityTitle", transaction.find("securityTitle").text.strip()])
            
        if transaction.find("transactionDate") is not None:
            infoTransaction.append(["transactionDate", transaction.find("transactionDate").text.strip()])
        
        if transaction.find("transactionCoding") is not None:    
            trnsctnCoding = transaction.find("transactionCoding")
            if trnsctnCoding.find("transactionCode") is not None:
                infoTransaction.append(["transactionCode", trnsctnCoding.find("transactionCode").text.strip()])
        
        if transaction.find("transactionAmounts") is not None:
            trnsctnAmounts = transaction.find("transactionAmounts")
            if trnsctnAmounts.find("transactionShares") is not None:
                infoTransaction.append(["transactionShares", trnsctnAmounts.find("transactionShares").text.strip()])
            if trnsctnAmounts.find("transactionPricePerShare") is not None:
                infoTransaction.append(["transactionPricePerShare", trnsctnAmounts.find("transactionPricePerShare").text.strip()])

        if transaction.find("sharesOwnedFollowingTransaction") is not None:
            infoTransaction.append(["sharesOwnedFollowingTransaction", transaction.find("sharesOwnedFollowingTransaction").text.strip()])
        if transaction.find("directOrIndirectOwnership") is not None:
            infoTransaction.append(["directOrIndirectOwnership", transaction.find("directOrIndirectOwnership").text.strip()])
        return(infoTransaction)
    
    def get_non_derivative_table(text):
        tableNonDerivative = text.find_all(re.compile(r"nonDerivativeTransaction|nonDerivativeHolding"))
        infoNonTable = list()
        for transaction in tableNonDerivative:
            transactionDict = dict()
            infoTransaction = get_transaction_row(transaction)
            for item in infoTransaction:
                transactionDict[item[0]] = item[1]
            
            infoNonTable.append(pd.DataFrame.from_dict([transactionDict]))
        if len(infoNonTable) > 0:
            data = pd.concat(infoNonTable, sort = False, ignore_index = True)
        else:
            data = pd.DataFrame()
        return(data)
    
    def standardize_columns(df):
        temp_df = df.copy()
        
        columns=['transactionDate', 'transactionCode', 'securityTitle',
        'transactionShares', 'transactionPricePerShare',
        'sharesOwnedFollowingTransaction', 'directOrIndirectOwnership',
        'accessionNumber', 'firmTicker', 'executiveName', 'isDirector', 'isOfficer', 'isOther', 'isTenPercentOwner',
        'officerTitle', 'remarks', 'periodOfReport', 'issuerCik', 'rptOwnerCik']

        for col in columns:
            if col not in temp_df.columns:
                temp_df[col] = None

        cols_rename = {'transactionDate': 'transaction_date',
                        'transactionCode':'transaction_code', 
                        'securityTitle': 'security_title',
                        'transactionShares': 'shares', 
                        'transactionPricePerShare': 'price_per_share', 
                        'sharesOwnedFollowingTransaction': 'remaining_shares', 
                        'directOrIndirectOwnership': 'direct_or_indirect_ownership',
                        'accessionNumber': 'accession_number',
                        'executiveName': 'name', 
                        'officerTitle': 'title', 
                        'remarks':'remarks', 
                        'isDirector': 'is_director',
                        'isOfficer': 'is_officer', 
                        'isOther': 'is_other', 
                        'isTenPercentOwner': 'is_ten_percent_owner',
                        'periodOfReport': 'reporting_date',
                        'issuerCik': 'issuer_cik',
                        'rptOwnerCik': 'owner_cik'}
        
        temp_df = temp_df.rename(columns=cols_rename)
        
        return temp_df[cols_rename.values()]
        
    
    def convert_to_boolean(value: str) -> bool:
        if value in ["1", "true"]:
            return True
        else:
            return False
        
    def clean_title(title: str = None,
                    remarks: str = None,
                    is_officer: bool = False,
                    is_director: bool = False,
                    is_other: bool = False,
                    is_ten_percent_owner: bool = False):
                
        if title:
            if 'see remarks' in title.lower():
                title = remarks
            return title

        title: str = ""
        if is_director:
            title = "Director"
        elif is_officer:
            title = "Officer"
        elif is_other:
            title = "Other"

        if is_ten_percent_owner:
            title = f"{title}, 10% Owner" if title else "10% Owner"
            
        return title
    
    def process_data(df):
        temp_df = df.copy()
        
        temp_df = standardize_columns(temp_df)
        
        temp_df = temp_df.query('transaction_code in ["P", "S"]')
                
        temp_df[['is_director', 'is_officer', 'is_other', 'is_ten_percent_owner']] = temp_df[['is_director', 'is_officer', 'is_other', 'is_ten_percent_owner']].map(convert_to_boolean)
        
        if not temp_df.empty:
            temp_df['title'] = temp_df.apply(lambda x: clean_title(x['title'], x['remarks'], x['is_officer'], x['is_director'], x['is_other'], x['is_ten_percent_owner']), axis = 1)
        
        transaction_code_map = {'P': 'Buy', 'S': 'Sell'}
        temp_df['buy_sell'] = temp_df['transaction_code'].map(transaction_code_map)

        ownership_map = {'D': 'Direct', 'I': 'Indirect'}
        temp_df['ownership_nature'] = temp_df['direct_or_indirect_ownership'].map(ownership_map)
        
        float_cols = ['shares', 'price_per_share', 'remaining_shares']
        temp_df[float_cols] = temp_df[float_cols].apply(pd.to_numeric, errors='coerce', downcast='float')
        
        int_cols = ['issuer_cik', 'owner_cik']
        temp_df[int_cols] = temp_df[int_cols].astype(int)

        str_cols = ['security_title', 'accession_number', 'name', 'title', 'reporting_date', 'buy_sell', 'ownership_nature']
        temp_df[str_cols] = temp_df[str_cols].astype(str)
        
        dt_cols = ['transaction_date', 'reporting_date']
        temp_df[dt_cols] = temp_df[dt_cols].apply(pd.to_datetime, errors='coerce', format='%Y-%m-%d')
        
        temp_df = temp_df.drop(columns = ['remarks', 'is_director', 'is_officer', 'is_other', 'is_ten_percent_owner', 'direct_or_indirect_ownership', 'transaction_code'])
        return temp_df
    
    soup = bs(xml_content, "xml")
    identityData = get_identity(soup)
    dataNonTable = get_non_derivative_table(soup)
    
    identityData["accessionNumber"] = accession_number
    dataNonTable["accessionNumber"] = accession_number
    
    data = pd.merge(dataNonTable, identityData, on = "accessionNumber")
    
    data = process_data(data)
    
    return(data)

def retrieve_company_form_4_filing(company_ticker, min_date, excluded_acc_num=[]):
    company_df = pd.DataFrame()
    company = Company(company_ticker)
    company_cik = company.cik
    filings = company.get_filings(form="4")
    for filing in filings:
        if filing.filing_date < min_date:
            break
        if filing.accession_number in excluded_acc_num:
            continue
        
        temp_df = parse_form_4_filing(filing.accession_number, filing.xml())
        temp_df['url'] = filing.primary_documents[0].url
        temp_df['filing_date'] = filing.filing_date
        temp_df['filing_date'] = temp_df['filing_date'].apply(pd.to_datetime, errors='coerce', format='%Y-%m-%d')
        
        if not temp_df.empty:
            company_df = pd.concat([company_df, temp_df], ignore_index=True)
            
    if not company_df.empty:  
        company_df = company_df.query('issuer_cik == @company_cik & transaction_date >= @min_date')
    
    return company_df

def retrieve_companies_form_4_filing(symbols, db_latest_form_4=pd.DataFrame()):
    def get_min_date_and_acc_num(symbol):
        if not db_latest_form_4.empty and symbol in db_latest_form_4.symbol.values:
            symbol_df = db_latest_form_4.query('symbol == @symbol').head(1)
            min_date = symbol_df.filing_date.values[0]
            acc_num = symbol_df.accession_numbers.values[0]
        else:
            min_date = date.today() - timedelta(days=365*2)
            acc_num = []
        return min_date, acc_num

    companies_df = pd.DataFrame()

    for symbol in symbols:
        min_date, acc_num = get_min_date_and_acc_num(symbol)
        try:
            temp_df = retrieve_company_form_4_filing(symbol, min_date, acc_num)
            if not temp_df.empty:
                companies_df = pd.concat([companies_df, temp_df], ignore_index=True)
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")

    return companies_df

def match_company_stock_row(row, db_company_stock):
    """
    Match the 'cik' in the 'db_company_stock' DataFrame with the 'issuer_cik' in the given row,
    and ensure that the 'security_title' in the given row is contained in the 'security_titles' of 'db_company_stock'.
    
    Parameters:
    - row (Series): A row from the DataFrame containing transaction data.
    - db_company_stock (DataFrame): DataFrame containing company stock data from the database.
    
    Returns:
    - company_stock_id (int or None): Matching company stock id or None if no match is found.
    """
    
    # Filter db_company_stock DataFrame based on cik
    filtered_stock = db_company_stock[db_company_stock['cik'] == row['issuer_cik']]
    
    # If any matching rows found
    if not filtered_stock.empty:
        # Get the 'security_titles' for the matching 'cik'
        valid_security_titles = filtered_stock['valid_security_titles'].values[0]
        security_title = row['security_title'].lower()
        
        if valid_security_titles is None:
            valid_security_titles = []
            
        invalid_security_titles = filtered_stock['invalid_security_titles'].values[0]
        if invalid_security_titles is None:
            invalid_security_titles = []
        
        if security_title in valid_security_titles:
            return filtered_stock['id'].values[0]
        
        elif security_title in invalid_security_titles:
            return np.nan
        
        else:
            return "-1"
        
    else:
        return np.nan  # Return None if no match is found

def process_company_stock_id(df, db_company_stock):
    """
    Process DataFrame 'df' to match company stock and update 'company_stock_id'.
    Save unlisted security titles to a CSV file if any are found.

    Parameters:
    - df (DataFrame): DataFrame containing transaction data.
    - db_company_stock (DataFrame): DataFrame containing company stock data from the database.

    Returns:
    - df (DataFrame): Updated DataFrame with 'company_stock_id' column.
    """
    temp_df = df.copy()
    
    # Match company stock and update 'company_stock_id'
    temp_df['company_stock_id'] = temp_df.apply(lambda row: match_company_stock_row(row, db_company_stock), axis=1)
    temp_df['company_stock_id'] = temp_df['company_stock_id'].astype(float)

    # Extract unlisted security titles
    unlisted_security_titles = temp_df[temp_df['company_stock_id'] == -1]

    # Save unlisted security titles to a CSV file if any are found
    if not unlisted_security_titles.empty:
        dt = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        unlisted_security_titles.to_csv(f'unlisted_security_titles_{dt}.csv', index=False)

    temp_df = temp_df[temp_df['company_stock_id'] > 0]
    temp_df = temp_df.sort_values(by='reporting_date', ascending=False)
    
    return temp_df

def create_sec_person_df(df, db_sec_person_ciks):
    sec_person_df = df.groupby('owner_cik').first().reset_index()
    sec_person_df = sec_person_df.rename(columns={'owner_cik': 'cik'})
    sec_person_df = sec_person_df[~sec_person_df['cik'].isin(db_sec_person_ciks)]
    return sec_person_df

def create_company_owner_df(df):
    company_owner_df = df.groupby(['owner_cik', 'company_stock_id']).first().reset_index()
    company_owner_df = company_owner_df.rename(columns={'owner_cik': 'person_cik', 'title':'position'})
    return company_owner_df

def retrieve_company_owner_id(row, db_company_owner):
    """
    Match the 'person_cik' and 'company_stock_id' in the 'db_company_owner' with the 'owner_cik' and 'company_stock_id' in the given row.
    
    Parameters:
    - row (Series): A row from the DataFrame containing transaction data.
    - db_company_owner (DataFrame): DataFrame containing company owner data from the database.
    
    Returns:
    - company_owner_id (int or None): Matching company owner id or None if no match is found.
    """
    # Filter db_company_owner based on owner_cik and company_stock_id
    filtered_owner = db_company_owner[(db_company_owner['person_cik'] == row['owner_cik']) & (db_company_owner['company_stock_id'] == row['company_stock_id'])]
    
    if not filtered_owner.empty:
        return filtered_owner['id'].values[0]
    else:
        return None
    
def create_form_4_filing_df(df, db_company_owner):
    df['company_owner_id'] = df.apply(lambda row: retrieve_company_owner_id(row, db_company_owner), axis=1)
    form_4_filing_df = df.groupby('accession_number').first().reset_index()
    return form_4_filing_df

def retrieve_filing_id(row, db_form_4_filing):
    """
    Match the 'accession_number' in the 'db_form_4_filing' with the 'accession_number' in the given row,
    
    Parameters:
    - row (Series): A row from the DataFrame containing transaction data.
    - db_form_4_filing (DataFrame): DataFrame containing form 4 filing data from the database.
    
    Returns:
    - filing_id (int or None): Matching filing id or None if no match is found.
    """
    # Filter db_form_4_filing based on accession_number
    filtered_filing = db_form_4_filing[db_form_4_filing['accession_number'] == row['accession_number']]
    
    if not filtered_filing.empty:
        return filtered_filing['id'].values[0]
    else:
        return None
    
if __name__ == "__main__":
    set_identity("MyCompanyName my.email@domain.com")

    load_dotenv()
    connection_string = os.getenv('DATABASE_URL')
    nc = NeonConnector(connection_string)
    
    response = nc.select_query("SELECT symbol from company_stock")
    symbols = [x['symbol'] for x in response]
    response = nc.select_query("SELECT * FROM get_latest_form_4()")
    db_latest_form_4 = pd.DataFrame(response)
    df = retrieve_companies_form_4_filing(symbols, db_latest_form_4)
    
    response = nc.select_query('SELECT * from company_stock')
    db_company_stock = pd.DataFrame(response)
    df = process_company_stock_id(df, db_company_stock)
    
    response = nc.select_query('SELECT cik from sec_person')
    db_sec_person_ciks = [row['cik'] for row in response]
    sec_person_df = create_sec_person_df(df, db_sec_person_ciks)
    sec_person_recs = nc.convert_df_to_records(sec_person_df[['cik', 'name']], int_cols=['cik'])
    if len(sec_person_recs) > 0:
        nc.batch_upsert('sec_person', sec_person_recs, conflict_columns=['cik'])
    
    company_owner_df = create_company_owner_df(df)
    company_owner_recs = nc.convert_df_to_records(company_owner_df[['person_cik', 'company_stock_id', 'position']])
    if len(company_owner_recs) > 0:
        nc.batch_upsert('company_owner', company_owner_recs, conflict_columns=['person_cik', 'company_stock_id'])
    
    response = nc.select_query('SELECT * FROM company_owner')
    db_company_owner = pd.DataFrame(response)
    form_4_filing_df = create_form_4_filing_df(df, db_company_owner)
    form_4_filing_recs = nc.convert_df_to_records(form_4_filing_df[['accession_number', 'reporting_date', 'company_owner_id']])
    if len(form_4_filing_recs) > 0:
        nc.batch_upsert('form_4_filing', form_4_filing_recs, conflict_columns=['accession_number'])
    
    response = nc.select_query('SELECT * FROM form_4_filing')
    db_form_4_filing = pd.DataFrame(response)
    df['filing_id'] = df.apply(lambda row: retrieve_filing_id(row, db_form_4_filing), axis=1)
    form_4_transaction_recs = nc.convert_df_to_records(df[['transaction_date', 'shares', 'price_per_share', 'buy_sell', 'ownership_nature','filing_id']])
    if len(form_4_transaction_recs) > 0:
        nc.batch_upsert('form_4_transaction', form_4_transaction_recs)
    
    
    