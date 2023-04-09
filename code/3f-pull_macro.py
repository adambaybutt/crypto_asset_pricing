import requests
import pandas_datareader as pdr
import pandas as pd

def pullMccrakenMacro(mccraken_url: str, mccraken_raw_fp: str) -> None:
    """
    Downloads the McCraken macroeconomic dataset from a given URL and saves it to a local file.

    Args:
        mccraken_url (str): The URL of the McCraken macroeconomic dataset.
        mccraken_raw_fp (str): The file path where the raw dataset will be saved.
        
    Returns:
        None
    """
    # Use requests library to get the contents of the dataset from the URL
    response = requests.get(mccraken_url)
    
    # Save the contents to a file
    with open(mccraken_raw_fp, "wb") as f:
        f.write(response.content)

def pullTRates(fred_names: list) -> pd.DataFrame:
    """
    Get Treasury rate data from FRED for a list of series names.
    
    Parameters:
        fred_names (list of str): List of series names on FRED.
        
    Returns:
        df (pandas.DataFrame): time series data containing the treasury data for each series.
    """
    # set the start date 
    start_date = '2014-01-01'

    # use the FRED API to get the data for each series name
    dfs = [pdr.DataReader(name, 'fred', start_date) for name in fred_names]

    # concatenate the data for all series into a single DataFrame
    df = pd.concat(dfs, axis=1)
    
    return df

def pullPolicyUncertainty() -> pd.DataFrame:
    """ Pull policy uncertainty data from their website for a set of urls.
    
    Args: None.
    
    Returns: (pd.DataFrame) time series dataframe with columns of uncertainty measures.
    """
    # set target urls
    gpu_url = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
    us_mpu_url = "https://www.policyuncertainty.com/media/US_MPU_Monthly.xlsx"
    twitter_url = "https://www.dropbox.com/s/o4ddj33odyyz4v6/Twitter_Economic_Uncertainty.xlsx?dl=1"
    vol_url = "https://www.policyuncertainty.com/media/EMV_Data.xlsx"

    # pull and clean gpu data
    response = requests.get(gpu_url)
    gpu_df = pd.read_excel(response.content, engine='openpyxl')
    gpu_df = gpu_df.dropna()
    gpu_df['date'] = pd.to_datetime(gpu_df[['Year', 'Month']].assign(day=1))
    gpu_df = gpu_df[['date', 'GEPU_ppp']]
    gpu_df = gpu_df.rename(columns={'GEPU_ppp': 'gepu'})

    # pull and clean us_mpu data
    response = requests.get(us_mpu_url)
    us_mpu_df = pd.read_excel(response.content, engine='openpyxl')
    us_mpu_df = us_mpu_df.dropna()
    us_mpu_df['date'] = pd.to_datetime(us_mpu_df[['Year', 'Month']].assign(day=1))
    us_mpu_df = us_mpu_df[['date', 'BBD MPU Index Based on Access World News']]
    us_mpu_df = us_mpu_df.rename(columns={'BBD MPU Index Based on Access World News': 'us_mpu'})

    # pull and clean twitter uncertainty data
    response = requests.get(twitter_url)
    teu_df = pd.read_excel(response.content, engine='openpyxl')
    teu_df = teu_df.dropna()
    teu_df = teu_df[['date', 'TEU-SCA', 'TMU-SCA']]

    # pull and clean volality data
    response = requests.get(vol_url)
    vol_df = pd.read_excel(response.content, engine='openpyxl')
    vol_df = vol_df.dropna()
    vol_df['date'] = pd.to_datetime(vol_df[['Year', 'Month']].assign(day=1))
    vol_df = vol_df[['date', 'Overall EMV Tracker', 'Macro – Inflation EMV Indicator']]
    vol_df = vol_df.rename(columns={'Overall EMV Tracker': 'emv',
                                    'Macro – Inflation EMV Indicator': 'emv_inflation'})

    # merge data together
    df = gpu_df.merge(us_mpu_df, on='date', how='outer', validate='one_to_one')
    df = df.merge(teu_df, on='date', how='outer', validate='one_to_one')
    df = df.merge(vol_df, on='date', how='outer', validate='one_to_one')

    return df

if __name__ == "__main__":
    # set args
    mccraken_url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/2023-01.csv"
    mccraken_raw_fp = '../data/raw/mccraken_macro.csv'
    fred_names = ['DGS1MO', 'DFII5', 'DFII7', 'DFII10', 'DFII20', 'DFII30']
    t_fp = '../data/raw/treasury_macro.pkl'
    u_fp    = '../data/raw/uncertainty_macro.pkl'
    
    # pull raw mccracken data
    mccraken_df = pullMccrakenMacro(mccraken_url, mccraken_raw_fp)

    # pull treasury rates
    t_df = pullTRates(fred_names)
    t_df.to_pickle(t_fp)

    # pull uncertainty data
    u_df = pullPolicyUncertainty()
    u_df.to_pickle(u_fp)

