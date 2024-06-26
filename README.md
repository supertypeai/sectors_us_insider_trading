Parse data from SEC Form 4 XML (e.g. https://www.sec.gov/Archives/edgar/data/2488/000000248822000108/xslF345X03/wf-form4_165420068129969.xml) and update them to the following tables: `sec_person`, `company_owner`, `form_4_filing` and `form_4_transaction`

## Columns: 
- transaction_date: stored in `form_4_transaction`
- buy_sell: stored in `form_4_transaction`
- shares: stored in `form_4_transaction`
- price_per_share: stored in `form_4_transaction`
- ownership_nature: stored in `form_4_transaction`
- accession_number: stored in `form_4_filing`
- reporting_date: stored in `form_4_filing`
- url: stored in `form_4_filing`
- filing_date: stored in `form_4_filing`
- security_title: to be matched with `valid_security_titles` and `invalid_security_titles` in `company_stock`
- issuer_cik: to be matched with `cik` in `company_stock`
- name (referring to officer): stored in `sec_person`
- owner_cik: stored in `sec_person`
- title (referring to officer): stored in `company_owner`

## Notes 
`valid_security_titles` and `invalid_security_titles` in `company_stock` should be added manually. If the security title in the Form 4 is not listed in either `valid_security_titles` or `invalid_security_titles`, then the row will be saved in a csv file `unlisted_security_titles_batch_n.csv`.
    
