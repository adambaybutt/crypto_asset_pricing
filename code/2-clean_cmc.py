import pandas as pd
import numpy as np

def cleanAssetMetadata(cw_df: pd.DataFrame, column_map: dict, dropna: bool = True) -> pd.DataFrame:
    """ Clean asset metadata to return cleaned dataframe. 

    Args:
        cw_df (pd.DataFrame): DataFrame containing asset metadata to be cleaned.
        column_map (dict): a mapping of the current column names to the desired column names.
        dropna (bool): whether to drop any rows with missing values in key columns.
    
    Returns:
        cw_df (pd.DataFrame): cleaned DataFrame.
    """
    # confirm has required columns
    for k, v in column_map.items():
        if k not in cw_df.columns:
            raise ValueError(f"Input DataFrame must contain '{k}' column.")
        
    # apply column map renaming
    cw_df = cw_df.rename(columns=column_map)

    # subset to useful columns
    cw_df = cw_df[['cmc_id', 'slug_cmc', 'symbol_cmc', 'first_date_cmc', 'last_date_cmc']]

    # convert date columns to date type
    cw_df['first_date_cmc'] = pd.to_datetime(cw_df.first_date_cmc, format='%Y-%m-%d', utc=False)
    cw_df['last_date_cmc'] = pd.to_datetime(cw_df.last_date_cmc, format='%Y-%m-%d', utc=False)

    # drop rows with missing values in key columns
    if dropna:
        cw_df = cw_df.dropna(subset=['cmc_id', 'slug_cmc'])
    
    # assert that each row has a unique `cmc_id` and `slug_cmc` value (if desired)
    if len(cw_df) != len(cw_df['cmc_id'].unique()):
        raise ValueError("Input DataFrame has non-unique 'cmc_id' values.")
    if len(cw_df) != len(cw_df['slug_cmc'].unique()):
        raise ValueError("Input DataFrame has non-unique 'slug_cmc' values.")
    
    # sort values and reset index
    cw_df = cw_df.sort_values(by='cmc_id', ignore_index=True)

    return cw_df
    
def cleanPanel(panel_df: pd.DataFrame, start_year: int=2015, end_date: str='2023-02-02', ) -> pd.DataFrame:
    """ clean panel of cmc prices, volume, and mcap data.
     
    Args:
        panel_df (pandas.DataFrame): panel data to clean.
        start_year (int): the minimum year to include in the DataFrame (default: 2015).
        end_date (str): the maximum date (inclusive) to include in the DataFrame (default: '2023-02-02').

    Returns:
        (pd.DataFrame): The cleaned DataFrame.
    """
    # confirm has the right columns
    expected_cols = ['date', 'cmc_id', 'usd_per_token', 'usd_mcap', 'usd_volume_24h']
    if not all(col in panel_df.columns for col in expected_cols):
        raise ValueError(f"Missing expected columns: {expected_cols}")
    
    # rename columns to standard convention (with data source name in it)
    panel_df = panel_df.rename(columns = {'usd_per_token': 'usd_per_token_cmc',
                                          'usd_mcap': 'usd_mcap_cmc',
                                          'usd_volume_24h': 'usd_volume_24h_cmc'})

    # convert columns to correct data type
    panel_df['date'] = pd.to_datetime(panel_df.date, format='%Y-%m-%d', utc=False)

    # set column order
    panel_df = panel_df[['date', 'cmc_id', 'usd_per_token_cmc', 'usd_mcap_cmc', 'usd_volume_24h_cmc']]

    # drop rows
    panel_df = panel_df[(panel_df.date.dt.year >= 2015) & (panel_df.date <= '2023-02-02')]
    panel_df = panel_df.dropna(how='any', subset=['date', 'cmc_id'])
    panel_df = panel_df.dropna(how='all', subset=['usd_per_token_cmc', 'usd_mcap_cmc', 'usd_volume_24h_cmc'])

    # form list of data columns to work with
    data_cols = list(panel_df.columns.values)
    data_cols.remove('date')
    data_cols.remove('cmc_id')

    # set negative values to missing and too large values to missing
    for col in data_cols:
        panel_df.loc[panel_df[col] < 0, col] = np.nan
        panel_df.loc[panel_df[col] > 2e12, col] = np.nan

    # drop duplicated rows across id columns
    panel_df = panel_df.drop_duplicates(subset=['date', 'cmc_id'])

    # sort values and reset index
    panel_df = panel_df.sort_values(by=['date', 'cmc_id'], 
                                    ignore_index=True)

    return panel_df
    
if __name__ == "__main__":
    # set args
    asset_in_fp = "../data/raw/cmc_token_universe.pkl"
    panel_in_fp = "../data/raw/cmc_price_volume_mcap_panel.pkl"
    cw_new_old_col_mapping  = {'cmc_symbol': 'symbol_cmc',
                               'cmc_slug': 'slug_cmc',
                               'cmc_first_date': 'first_date_cmc',
                               'cmc_last_date': 'last_date_cmc'}
    cw_out_fp = "../data/derived/cmc_token_universe.pkl"
    panel_out_fp = "../data/derived/cmc_price_volume_mcap_panel.pkl"

    # import data
    cw_df = pd.read_pickle(asset_in_fp)
    panel_df = pd.read_pickle(panel_in_fp)

    # clean the data
    cw_df = cleanAssetMetadata(cw_df, cw_new_old_col_mapping)
    panel_df = cleanPanel(panel_df)

    # save the data
    cw_df.to_pickle(cw_out_fp)
    panel_df.to_pickle(panel_out_fp)
