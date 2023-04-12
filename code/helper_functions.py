import requests
import time
import logging
from typing import Any, Dict, Optional, List
import random
from datetime import datetime, timedelta
import pandas as pd
from itertools import chain
import numpy as np
logger = logging.getLogger(__name__)

class Helper:
    """ Class of helper functions specific to crypto asset pricing project. """

    @staticmethod
    def makeApiCall(url: str, headers: Optional[Dict[str, str]], params: Optional[Dict[str, str]] = None, 
                    retries: int = 5, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        Makes an API call to the given endpoint with the given parameters.

        Args:
        - url (str): string representing the URL for the API.
        - headers (Optional[Dict[str, str]]): dictionary containing the headers for the API call.
        - params (Optional[Dict[str, str]]): dictionary containing the parameters for the API call.
        - retries (int): integer representing the number of times to retry the API call in case of an error. Defaults to 4.
        - timeout (int): integer representing the timeout in seconds for the API call. Defaults to 5.
        
        Returns:
        - response (Optional[Dict[str, Any]]): the data from the API response, or None if the API call failed.
        """
        if params is None:
            params = {}
        if headers is None:
            headers = {}

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=8)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    logger.warning(f'The API call failed with error: {str(e)}')
                    logger.warning('403 Forbidden Error - Will not retry')
                    return None
                else:
                    logger.warning(f'The API call failed with error: {str(e)}')
            except requests.exceptions.RequestException as e:
                logger.warning(f'The API call failed with error: {str(e)}')
            if attempt == retries - 1:
                logger.error(f'The API call failed after {retries} attempts.')
                return None
            else:
                sleep_time = (4 ** attempt) * (0.5 + random.uniform(0, 1))
                logger.warning(f'Retrying after {sleep_time:.2f} seconds.')
                time.sleep(sleep_time)

    @staticmethod
    def generateYearlyCalendarYearDateList(time_start: str, time_end: str) -> List[str]:
        """
        Generates a list of dates in the format '%Y-%m-%d' representing the first day of each year between 
        the given start and end dates (inclusive).

        Args:
        - time_start (str): start date in the format '%Y-%m-%d'.
        - time_end (str): end date in the format '%Y-%m-%d'.

        Returns:
        - List[str]: a list of dates in the format '%Y-%m-%d' representing the first day of each year 
                     between the given start and end dates (inclusive).
        """
        # Convert start and end dates to datetime objects
        start_date = datetime.strptime(time_start, '%Y-%m-%d')
        end_date = datetime.strptime(time_end, '%Y-%m-%d')

        # Calculate the difference between the years of the start and end dates
        year_diff = end_date.year - start_date.year

        # Generate list of dates based on the difference in years
        if year_diff == 0:
            dates = [time_start, time_end]
        elif year_diff > 0:
            dates = [time_start]
            for i in range(year_diff):
                dates.append(datetime(start_date.year + 1 + i, 1, 1).strftime('%Y-%m-%d'))
            dates.append(time_end)

        return dates
    
    @staticmethod
    def generateDailyDateList(start_time: str, end_time: str) -> List[str]:
        """ Generates a list of all days between the given start and end dates, inclusive.

        Args:
        - time_start (str): start date in the format '%Y-%m-%d'.
        - time_end (str): end date in the format '%Y-%m-%d'.

        Returns: 
        - List[str]: a list of dates in the format '%Y-%m-%d' representing all days including
                    and in between the given two dates.
        """
        date_list = []
        current_date = datetime.strptime(start_time, '%Y-%m-%d')
        end_date = datetime.strptime(end_time, '%Y-%m-%d')
        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        return date_list

    @staticmethod
    def findUniqueAssets(asset_universe_dict: Dict[str, List[str]]) -> List[str]:
        """
        Determine the unique assets in the universe and return them as a sorted list.

        Args:
            asset_universe_dict (dict): A dictionary where the keys are the first date of each month
                in the study period and and the values are lists of assets for that month.

        Returns:
            list: A sorted list of unique assets.
        """
        # Flatten the lists of assets and create a set to ensure uniqueness.
        unique_assets = set(chain.from_iterable(asset_universe_dict.values()))

        # Convert the set to a list and sort it.
        sorted_unique_assets = sorted(list(unique_assets))

        return sorted_unique_assets
    
    @staticmethod
    def xsecNormalizeToMinusOneOne(df: pd.DataFrame, target_col: str, asset_col: str) -> pd.DataFrame:
        """
        Normalize the target_col within each date in the input DataFrame, creating a new
        column target_col+`_norm` with values equally spaced between -1 and 1.

        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame with columns: date, asset_cg, target_col, and others.
        target_col: str
            Name of the target column to normalize.
        asset_col: str
            Name of the column containing the names of the assets.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with a new column containing the normalized values 
            of target_col within each date.
        """
        # randomly sort rows before the sort by target col to ties are randomly sorted
        df = df.sample(frac=1).reset_index(drop=True)

        # sort df by date and target col
        df = df.sort_values(by=['date', target_col], ignore_index=True)

        # define a custom function to add linearly spaced values within each group
        def addLinspace(group):
            n_rows = len(group)
            group[target_col+'_norm'] = np.linspace(-1, 1, n_rows)
            return group

        # add the linearly spaced column between -1 and 1 within each date of the DataFrame
        df = df.groupby('date', group_keys=True).apply(addLinspace).reset_index(drop=True)

        # clean up by resorting
        df = df.sort_values(by=['date', asset_col], ignore_index=True)
        
        return df

class CoinAPI:
    """ Class of helper functions specific to working with CoinAPI."""

    @staticmethod
    def pullMarketInfo(base_url: str, base_headers: Dict[str, str], target_exchanges: List[str], target_assets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame containing information about coinapi markets that are on a target exchange with
            USD or stablecoin quote asset.

        Args:
            base_url: A string representing the base URL of the CoinAPI service.
            base_headers: A dictionary representing the headers to be sent with the API request.
            target_exchanges: A list of strings with the target exchanges for this study.
            target_assets: A list of strings with the target assets for this study.

        Returns:
            A Pandas DataFrame containing information about cryptocurrency markets.
        """
        # Build target URL
        target_url = 'symbols'
        url        = f"{base_url}{target_url}"
        headers    = base_headers.copy()

        # Call API and convert to DataFrame
        response_json = Helper.makeApiCall(url, headers=headers)
        df = pd.DataFrame(response_json)

        # subset to exchanges of interest
        df = df[df.exchange_id.isin(target_exchanges)]

        # clean columns
        df['data_start'] = pd.to_datetime(df.data_start)
        df['data_end'] = pd.to_datetime(df.data_end)
        df['duration_days'] = (df.data_end - df.data_start).dt.days

        # subset to assets of interest
        df = df[df.symbol_type=='SPOT'] # spot markets
        df = df[df.asset_id_quote.isin(['USD', 'USDC', 'USDT'])] # quote asset is fiat USD or stablecoin USD
        df = df.dropna(subset=['data_start', 'data_end'])  # have data
        df = df[df.duration_days > 120] # have at least four months of data
        target_date = pd.Timestamp('2022-09-01')
        df = df[df.data_start <= target_date] # have at least four months of data in target window

        # remove symbols that are derivatives of other symbols or stablecoins
        assets_to_remove = ['WBTC', 'WLUNA', 'WNXM', 'TBTC', 'CUSD', 'MUSD', 'NUSD', 'DAI', 'BUSD', 'CUSDT', 
            'GUSD', 'LUSD', 'OUSD', 'USDJ', 'USDK', 'USDN', 'USDT', 'USDC', 'AOA', 'AUSD', 'ERN', 'KRW', 'MTL', 
            'TUSD', 'SUSD', 'USDD', 'UST', 'USTC', 'EUR', 'AUD', 'GBP', 'CAD', 'CBETH', 'LBP', 'SOS']
        df = df[~df.asset_id_base.isin(assets_to_remove)]
        df = df[~df.asset_id_base.str.contains('3L|3S')]

        # subset to target assets if list is given
        if target_assets is not None:
            df = df[df.asset_id_base.isin(target_assets)]

        return df.reset_index(drop=True)

    @staticmethod
    def pullExchangeRates(base_url: str, base_headers: Dict[str, str], target_assets: List[str],
                          target_freq: str, time_start: str, time_end: str) -> pd.DataFrame:
        """
        Returns a DataFrame containing prices of usdc and usdt.

        Args:
            base_url: A string representing the base URL of the CoinAPI service.
            base_headers: A dictionary representing the headers to be sent with the API request.
            target_assets: A list of strings of assets included in this study.
            target_freq: A string of the frequency to pull data for; either '1DAY' or '1HRS'.
            time_start: A string of format '%Y-%m-%d' of the date for the study start.
            time_end: A string of format '%Y-%m-%d' of the date for the study end.

        Returns:
            A Pandas DataFrame containing usdt and usdc price timeserieses.
        """
        # set params
        assert target_freq in ['1HRS', '1DAY']
        headers = base_headers.copy()
        params = {'period_id': target_freq, 'limit': 10000}

        # initialize data frame for the results
        df = pd.DataFrame()

        # loop over assets to pull
        for i in range(len(target_assets)):
            # update asset to pull
            asset = target_assets[i]

            # report how many assets completed
            print(f"Processing asset #{i+1} ({(i+1)/len(target_assets)*100:.2f}%): {asset}")

            # make the call
            url = url = f"{base_url}exchangerate/{asset}/USD/history"
            if target_freq == '1DAY': # if the request is at daily level then make it
                params['time_start'] = time_start
                params['time_end'] = time_end
                response_json = Helper.makeApiCall(url, headers=headers, params=params)
                asset_df = pd.DataFrame(response_json)
            elif target_freq == '1HRS': # if at hourly level break into calendar year requests and append
                asset_df = pd.DataFrame()
                time_list = Helper.generateYearlyCalendarYearDateList(time_start, time_end)
                for j in range(len(time_list)-1):
                    params['time_start'] = time_list[j]
                    params['time_end']   = time_list[j+1]
                    response_json = Helper.makeApiCall(url, headers=headers, params=params)
                    temp_df = pd.DataFrame(response_json)
                    asset_df = pd.concat((asset_df, temp_df))

            # clean the exchange rate df for the given asset
            asset_df = asset_df[asset_df.rate_close!=0] # remove invalid prices
            asset_df = asset_df[asset_df.time_period_end.str[:2]=='20'] # remove broken dates
            asset_df['date'] = pd.to_datetime(asset_df.time_period_end, utc=True).dt.tz_localize(None)
            asset_df['usd_per_token_ref'] = asset_df.rate_close
            asset_df = asset_df[['date', 'usd_per_token_ref']]

            # ensure asset data is present for all dates 
            asset_df.set_index('date', inplace=True)
            if target_freq == '1DAY':
                date_range = pd.date_range(start=asset_df.index.min(), end=asset_df.index.max(), freq='D')
            elif target_freq == '1HRS':
                date_range = pd.date_range(start=asset_df.index.min(), end=asset_df.index.max(), freq='H')
            asset_df = asset_df.reindex(date_range)
            asset_df['usd_per_token_ref'] = asset_df.usd_per_token_ref.ffill()

            # ensure stablecoins are in valid range 
            if asset in ['USDC', 'USDT']:
                asset_df.loc[asset_df.usd_per_token_ref>2, 'usd_per_token_ref'] = np.nan
                asset_df.loc[asset_df.usd_per_token_ref<0.5, 'usd_per_token_ref'] = np.nan
                asset_df['usd_per_token_ref'] = asset_df.usd_per_token_ref.ffill()

            # final clean
            asset_df = asset_df.reset_index()
            asset_df = asset_df.rename(columns={'index': 'date'})
            asset_df['asset'] = asset
            asset_df = asset_df[['date', 'asset', 'usd_per_token_ref']]

            # ensure no missing values
            assert 0 == asset_df.isnull().sum().sum()

            # append
            df = pd.concat((df, asset_df))

        return df.sort_values(by='date', ignore_index=True)

    @staticmethod
    def seperateMacroAndAssetRefPrices(prices_df: pd.DataFrame) -> tuple:
        """ Separate macro and asset reference prices from a given DataFrame.

        Args:
            prices_df: A pandas DataFrame containing asset prices and asset types.

        Returns:
            A tuple of two pandas DataFrames containing macro reference prices and
            asset reference prices respectively.
        """
        # Separate stablecoin DataFrames and remaining asset DataFrame
        usdc_df = prices_df[prices_df.asset=='USDC']
        usdt_df = prices_df[prices_df.asset=='USDT']
        prices_df = prices_df[~prices_df.asset.isin(['USDT', 'USDC'])]

        # Rename columns and drop 'asset' column from stablecoin DataFrames
        usdc_df = usdc_df.rename(columns={'usd_per_token_ref': 'usd_per_usdc'})
        usdt_df = usdt_df.rename(columns={'usd_per_token_ref': 'usd_per_usdt'})
        usdc_df = usdc_df.drop(columns=['asset'], axis=1)
        usdt_df = usdt_df.drop(columns=['asset'], axis=1)

        # Merge stablecoins into a macro DataFrame
        macro_df = usdc_df.merge(usdt_df, on='date', how='outer', validate='one_to_one')

        # Sort DataFrames by date and reset index
        prices_df = prices_df.sort_values(by=['date', 'asset'], ignore_index=True)
        macro_df = macro_df.sort_values(by='date', ignore_index=True)

        return macro_df, prices_df

    @staticmethod
    def pullMarketData(base_url: str, base_headers: Dict[str, str], 
                       markets_list: List[str], target_freq: str, time_start: str, time_end: str) -> pd.DataFrame:
        """
        Returns a panel DataFrame containing market, datetime, prices, volumes, and trade counts.

        Args:
            base_url: A string representing the base URL of the CoinAPI service.
            base_headers: A dictionary representing the headers to be sent with the API request.
            markets_list: A list of strings of market names to pull.
            target_freq: A string of the frequency to pull data for; either '1DAY' or '1HRS'.
            time_start: A string of format '%Y-%m-%d' of the date for the study start.
            time_end: A string of format '%Y-%m-%d' of the date for the study end.

        Returns:
            A Pandas DataFrame panel of dates and markets with their prices, volumes, and trade count
        """
        # set params
        assert target_freq in ['1HRS', '1DAY']
        headers = base_headers.copy()
        params = {'period_id': target_freq, 'limit': 10000}

        # initialize data frame for the results
        df = pd.DataFrame()
        
        # pull all markets
        for i in range(len(markets_list)):
            # update market to pull
            market = markets_list[i]

            # monitor progress
            print(f"Processing market #{i+1} ({(i+1)/len(markets_list)*100:.2f}%): {market}")

            # make the call
            market_df = pd.DataFrame()
            url = f"{base_url}ohlcv/{market}/history"
            if target_freq == '1DAY': # if the request is at daily level then make it
                params['time_start'] = time_start
                params['time_end'] = time_end
                response_json = Helper.makeApiCall(url, headers=headers, params=params)
                market_df = pd.DataFrame(response_json)
            elif target_freq == '1HRS': # if at hourly level break into calendar year requests and append
                asset_df = pd.DataFrame()
                time_list = Helper.generateYearlyCalendarYearDateList(time_start, time_end)
                for j in range(len(time_list)-1):
                    params['time_start'] = time_list[j]
                    params['time_end']   = time_list[j+1]
                    response_json = Helper.makeApiCall(url, headers=headers, params=params)
                    temp_df = pd.DataFrame(response_json)
                    market_df = pd.concat((market_df, temp_df))

            # clean the data
            try:
                # clean the market_df
                market_df['symbol_id'] = market
                market_df = market_df[['symbol_id', 'time_period_end', 'price_close', 'volume_traded', 'trades_count']]

                # save data
                df = pd.concat((df, market_df))
            except:
                print(f"{market} did not have data")
                continue

        return df

    @staticmethod
    def cleanPanel(df: pd.DataFrame, prices_df: pd.DataFrame, 
                macro_df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """ clean the panel to return the coinapi prices and volumes.

        Args:
            df: Pandas DataFrame of raw date, asset, price, volume, and trade count.
            prices_df: Pandas DataFrame of date, asset, and reference price.
            macro_df: Pandas DataFrame of date and prices for USDC and USDT.
            target_freq: a string for the target frequency that we are working at.

        Returns:
            df: Pandas DataFrame of date, asset_id, usd_per_token, usd_volume, 
                trades_coinapi, and usd_per_token_ref.    
        """
        # remove broken dates
        df = df[df.time_period_end.str[:2]=='20']

        # remove asset-dates where there is a missing price and zero volume
        df = df[~(df.price_close.isnull() & (df.volume_traded==0) & (df.trades_count==0))]

        # extract names of exchange, base asset, and quote asset
        df['exchange'] = df['symbol_id'].str.split('_', n=4, expand=True)[0]
        df['asset_id'] = df['symbol_id'].str.split('_', n=4, expand=True)[2]
        df['quote_id'] = df['symbol_id'].str.split('_', n=4, expand=True)[3]
        df = df.drop(columns='symbol_id', axis=1)

        # form the date column
        df['date'] = pd.to_datetime(df.time_period_end, utc=True).dt.tz_localize(None)
        df = df.drop(columns='time_period_end', axis=1)

        # drop duplicates
        df = df.drop_duplicates(subset=['date', 'asset_id'])

        # merge on usdt and usdc prices
        df = df.merge(macro_df, on='date', how='left', validate='many_to_one')

        # form the price column
        df.loc[df.quote_id=='USD', 'usd_per_token_coinapi'] = df.loc[df.quote_id=='USD', 'price_close']
        df.loc[df.quote_id=='USDC', 'usd_per_token_coinapi'] = df.loc[df.quote_id=='USDC', 'price_close']*df.loc[df.quote_id=='USDC', 'usd_per_usdc']
        df.loc[df.quote_id=='USDT', 'usd_per_token_coinapi'] = df.loc[df.quote_id=='USDT', 'price_close']*df.loc[df.quote_id=='USDT', 'usd_per_usdt']
        assert 0 == df.usd_per_token_coinapi.isnull().sum()
        df = df.drop(columns=['price_close', 'usd_per_usdc', 'usd_per_usdt'])

        # form volume column
        df['usd_volume_coinapi'] = df.volume_traded*df.usd_per_token_coinapi
        assert 0 == df.usd_volume_coinapi.isnull().sum()

        # collapse to the asset date level
        df.loc[df.usd_volume_coinapi==0, 'usd_volume_coinapi'] = 1
        grouped = df.groupby(['date', 'asset_id'])
        weighted_avg = grouped.apply(lambda x: (x['usd_per_token_coinapi'] * x['usd_volume_coinapi']).sum() / x['usd_volume_coinapi'].sum())
        total_volume = grouped['usd_volume_coinapi'].sum()
        total_trades = grouped['trades_count'].sum()
        df = pd.DataFrame({'usd_per_token_coinapi': weighted_avg, 
                            'usd_volume_coinapi': total_volume, 
                            'trades_count': total_trades}).reset_index()
        df.loc[df.usd_volume_coinapi==1, 'usd_volume_coinapi'] = 0

        # check for valid ranges and dtypes
        assert 0 == df.usd_per_token_coinapi.isnull().sum()
        assert 0 == df.usd_volume_coinapi.isnull().sum()
        df = df[(df['usd_per_token_coinapi'] >= 0) & (df['usd_per_token_coinapi'] < 1e9)]
        df = df[(df['usd_volume_coinapi'] >= 0) & (df['usd_volume_coinapi'] < 5e11)]
        df = df[(df['trades_count'] >= 0) & (df['trades_count'] < 1e9)]

        # ensure dtypes are set
        df['usd_per_token_coinapi'] = df['usd_per_token_coinapi'].astype('float32')
        df['usd_volume_coinapi'] = df['usd_volume_coinapi'].astype('float32')
        df['trades_count'] = df['trades_count'].astype('float32')

        # ensure panel is sorted
        df = df.sort_values(by=['date', 'asset_id'], ignore_index=True)

        # initialize a new df
        final_df = pd.DataFrame(data={'date': [], 'asset_id': [], 'usd_per_token_coinapi': [], 'usd_volume_coinapi': [], 'trades_count': []})

        # loop over all assets to add missing days
        assets = list(np.unique(df.asset_id.values))
        for asset in assets:
            # subset to asset of interest
            asset_df = df[df.asset_id==asset].copy()

            # determine the date gaps
            date_gaps = []
            dates = asset_df.date.values
            for i in range(1, len(dates)):
                if target_freq == '1DAY':
                    date_gaps.append(np.timedelta64(dates[i]-dates[i-1], 'D').astype(int))
                elif target_freq == '1HRS':
                    date_gaps.append(np.timedelta64(dates[i]-dates[i-1], 'h').astype(int))

            # determine new days to add
            if target_freq == '1DAY':
                indices_to_expand = [i for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32)]
                num_datetime_to_add = [date_gaps[i] for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32)]
                start_datetimes = dates[indices_to_expand]
                new_datetimes = []
                for i in range(len(start_datetimes)):
                    start_datetime = start_datetimes[i]
                    datetime_to_add = num_datetime_to_add[i]
                    for j in range(1, datetime_to_add):
                        new_datetimes.append(start_datetime+np.timedelta64(24*(j), 'h'))
            elif target_freq == '1HRS':
                indices_to_expand = [i for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32*24)]
                num_datetime_to_add = [date_gaps[i] for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32*24)]
                start_datetimes = dates[indices_to_expand]
                new_datetimes = []
                for i in range(len(start_datetimes)):
                    start_datetime = start_datetimes[i]
                    datetime_to_add = num_datetime_to_add[i]
                    for j in range(1, datetime_to_add):
                        new_datetimes.append(start_datetime+np.timedelta64(j, 'h'))

            # add the new days to the asset df
            new_asset_df = pd.DataFrame(data={'date': new_datetimes})
            new_asset_df['asset_id'] = asset
            asset_df = pd.concat((asset_df, new_asset_df))
            asset_df = asset_df.sort_values(by='date', ignore_index=True)

            # forward fill the price column
            asset_df['usd_per_token_coinapi'] = asset_df.usd_per_token_coinapi.ffill()

            # replace volume and trades with zeros
            asset_df.loc[asset_df.usd_volume_coinapi.isnull(), 'usd_volume_coinapi'] = 0
            asset_df.loc[asset_df.trades_count.isnull(), 'trades_count'] = 0

            # add data to master df
            final_df = pd.concat((final_df, asset_df))

        # final clean
        del df
        df = final_df.copy()
        df = df.rename(columns={'trades_count': 'trades_coinapi'})
        df = df.sort_values(by=['date', 'asset_id'], ignore_index=True)
        assert not df.duplicated(subset=['date', 'asset_id']).any()
        assert 0 == df.isnull().sum().sum()

        # merge on the other prices
        if prices_df is not None:
            prices_df = prices_df.rename(columns={'asset': 'asset_id'})
            df = df.merge(prices_df, on=['date', 'asset_id'], how='left', validate='one_to_one')

        return df
    
class Coinmetrics:
    """ Class of helper functions specific to working with Coinmetrics API."""

    @staticmethod
    def pullAssetInfo(base_url: str, base_params: dict, target_assets: List[str]=[]) -> pd.DataFrame:
        """
        Returns a DataFrame containing information about cryptocurrency assets.

        Args:
            base_url: A string representing the base URL of the Coinmetrics service.
            base_params: A dictionary representing the parameters to be sent with the API request.

        Returns:
            A Pandas DataFrame containing information about cryptocurrency assets.
        """
        # Build target URL and headers
        target_url = "catalog-all/assets"
        url        = f"{base_url}{target_url}"
        params = base_params.copy()

        # Call API and convert to DataFrame
        response_json = Helper.makeApiCall(url, headers={}, params=params)
        df = pd.DataFrame(response_json['data'])

        # Subset to assets with trading data
        df = df[~df.markets.isnull()]

        # Subset to target assets if given
        if len(target_assets) > 0:
            df = df[df.asset.isin(target_assets)]
        
        # return asset info relevant columns and clean up index
        return df[['asset', 'full_name', 'exchanges', 'markets', 'metrics']].reset_index(drop=True)
    
    @staticmethod
    def pullExchangeInfo(base_url: str, base_params: dict, target_exchanges: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame containing information about cryptocurrency exchanges.

        Args:
            base_url (str): A string representing the base URL of the Coinmetrics service.
            base_params (dict): A dictionary representing the parameters to be sent with the API request.
            target_exchanges (List[str]): A list of strings with the target exchanges for this study.

        Returns:
            A Pandas DataFrame containing information about cryptocurrency exchanges.
        """
        # Build target URL and headers
        target_url = "catalog-all/exchanges"
        url        = f"{base_url}{target_url}"
        params = base_params.copy()

        # Call API and convert to DataFrame
        response_json = Helper.makeApiCall(url, headers={}, params=params)
        df = pd.DataFrame(response_json['data'])

        # Subset to relevant exchanges
        df = df[df.exchange.isin(target_exchanges)].reset_index(drop=True)
        
        return df
    
    @staticmethod
    def pullAndFormRelevantMarkets(exchanges_df: pd.DataFrame, assets_df: pd.DataFrame,
        base_url: str, base_params: Dict[str, str], target_assets: List[str]=[]) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing information about coinmetrics markets 
        that are on a target exchange with USD or stablecoin quote asset.

        Args:
        - exchanges_df (pd.DataFrame): a DataFrame containing exchange data
        - assets_df (pd.DataFrame): a DataFrame containing asset data
        - base_url (str): a string containing the base url for the API
        - base_params (Dict[str, str]): a dictionary containing the base parameters for the API

        Returns:
        - pd.DataFrame: a pandas DataFrame containing information about relevant markets
        """
        # form dataframe of all markets
        markets_list = []
        for markets in list(exchanges_df.markets.values):
            markets_list.extend(markets)
        df = pd.DataFrame(data={'market': markets_list})

        # remove duplicates
        df = df.drop_duplicates(subset='market')

        # form market info
        df['exchange'] = df['market'].str.split('-', n=4, expand=True)[0]
        df['asset'] = df['market'].str.split('-', n=4, expand=True)[1]
        df['quote'] = df['market'].str.split('-', n=4, expand=True)[2]
        df['type'] = df['market'].str.split('-', n=4, expand=True)[3]

        # subset to spot markets
        df = df[df.type == 'spot']
        df = df.drop(columns='type', axis=1)

        # subset to quote asset is USD, USDC, or USDT
        df = df[df.quote.isin(['usd', 'usdt', 'usdc'])]

        # if no asset universe given, then subset broadly:
        if len(target_assets) == 0:
            # remove assets that are derivatives of other symbols or stablecoins
            assets_to_remove = ['wbtc', 'wluna', 'wnxm', 'tbtc', 'cusd', 'musd', 'nusd', 'dai', 'busd', 
                                'cusdt', 'gusd', 'lusd', 'ousd', 'usdj', 'usdk', 'usdn', 'usdt', 'usdc', 
                                'aoa', 'ausd', 'ern', 'krw', 'mtl', 'tusd', 'susd', 'usdd', 'ust', 'ustc', 
                                'eur', 'aud', 'gbp', 'cad', 'cbeth', 'lbp', 'sos', 'usdp', '00', 'bifi_beef', 
                                'bifi_bifr', 'btcauction', 'cix100']
            df = df[~df.asset.isin(assets_to_remove)]
            df = df[~df['asset'].str.contains('3l|3s|2s|2l')]

            # remove assets if they have no coinmetrics metrics
            df = df[df.asset.isin(list(assets_df[~assets_df.metrics.isnull()].asset.values))]
        elif len(target_assets) > 0:
            df = df[df.asset.isin(target_assets)]

        # build target url and headers for call for market meta data
        target_url = "catalog-all/markets"
        url        = f"{base_url}{target_url}"
        params = base_params.copy()

        # call API and convert to DataFrame
        response_json = Helper.makeApiCall(url, headers={}, params=params)
        markets_df = pd.DataFrame(response_json['data'])

        # subset to markets and columnns of interest
        markets_df = markets_df[markets_df.market.isin(df.market.values)].reset_index(drop=True)
        markets_df = markets_df[['market', 'exchange', 'base', 'quote', 'quotes']]
        markets_df = markets_df.drop('quotes', axis=1).join(pd.json_normalize(markets_df.quotes))

        # drop markets without quote data
        markets_df = markets_df[~markets_df.min_time.isnull()]

        return markets_df 

    @staticmethod
    def pullAssetMetrics(base_url: str, base_params: Dict[str, str], assets_df: pd.DataFrame,
        study_start: str, study_end: str, target_freq: str, target_assets: List[str], target_metrics: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame containing Coinmetrics reference exchange rates for target assets.

        Args:
            base_url (str): Base URL for the API.
            base_params (Dict[str, str]): Base parameters for the API.
            assets_df (pd.DataFrame): DataFrame containing information about assets metrics.
            study_start (str): string time for the start of the study window in format 'YYYY-MM-DD'.
            study_end (str): string time for the end of the study window in format 'YYYY-MM-DD'.
            target_freq (str): the target frequency to study.
            target_assets (List[str]): List of target assets to pull reference rates for.
            target_metrics (List[str]): List of asset metrics to pull.

        Returns:
            pd.DataFrame: A pandas DataFrame containing price timeserieses for all target assets.
        """
        # Initialize DataFrame to return results
        df = pd.DataFrame()

        # Build API URL
        api_endpoint = 'timeseries/asset-metrics'
        url = f"{base_url}{api_endpoint}"

        # Define API parameters
        api_params = {
            'page_size': 10000,
            'limit_per_asset': 10000,
            'end_inclusive': False
        }

        # Merge base parameters with API parameters
        params = {**base_params, **api_params}

        # Loop over every asset
        for i in range(len(target_assets)):
            # update asset
            asset = target_assets[i]

            # monitor progress
            print(f"Processing the {i+1}th asset ({(i+1)/len(target_assets)*100:.2f}%): {asset}")

            # initialize object for this asset results
            asset_df = pd.DataFrame(data={'asset': [], 'time': []})

            # update params for this asset
            params['assets'] = asset

            # determine metrics to pull
            metrics_list_dict = assets_df[assets_df.asset==asset].metrics.values[0]
            if type(metrics_list_dict) is not list: # skip if no metrics are available
                continue
            metrics_df = pd.DataFrame(metrics_list_dict)
            metrics_df = metrics_df[metrics_df.metric.isin(target_metrics)]

            # Loop over every metrics to pull
            assert metrics_df.metric.is_unique
            metrics_list = metrics_df.metric.values
            for metric in metrics_list:
                # update params with this metric
                params['metrics'] = metric

                # form dataframe of different freq options for this metric
                metric_options_df = pd.DataFrame(metrics_df[metrics_df.metric==metric].frequencies.values[0])

                # set frequency and extract its min and max time
                freq_options = list(metric_options_df[metric_options_df.frequency.isin(['1d', '1h'])].frequency.values)
                if len(freq_options) == 0:
                    print(f"For {asset} when pulling {metric} neither 1d nor 1h is available.")
                    continue
                elif len(freq_options) > 0:
                    if (target_freq == '1h') & (target_freq in freq_options):
                        params['frequency'] = target_freq
                        min_time = metric_options_df[metric_options_df.frequency==target_freq].min_time.values[0]
                        max_time = metric_options_df[metric_options_df.frequency==target_freq].max_time.values[0]
                    elif (target_freq == '1h') & ('1h' not in freq_options) & ('1d' in freq_options):
                        params['frequency'] = '1d'
                        min_time = metric_options_df[metric_options_df.frequency=='1d'].min_time.values[0]
                        max_time = metric_options_df[metric_options_df.frequency=='1d'].max_time.values[0]
                    elif (target_freq == '1d') & (target_freq in freq_options):
                        params['frequency'] = target_freq
                        min_time = metric_options_df[metric_options_df.frequency==target_freq].min_time.values[0]
                        max_time = metric_options_df[metric_options_df.frequency==target_freq].max_time.values[0]
                    else:
                        continue

                # set the start and end date for this metric
                metric_start_dt = datetime.strptime(min_time[:10], '%Y-%m-%d')
                metric_end_dt = datetime.strptime(max_time[:10], '%Y-%m-%d')
                study_start_dt = datetime.strptime(study_start, '%Y-%m-%d')
                study_end_dt = datetime.strptime(study_end, '%Y-%m-%d')
                if study_start_dt >= metric_start_dt:
                    start_time = study_start_dt.strftime('%Y-%m-%d')
                else:
                    start_time = metric_start_dt.strftime('%Y-%m-%d')
                if study_end_dt <= metric_end_dt:
                    end_time = study_end_dt.strftime('%Y-%m-%d')
                else:
                    end_time = metric_end_dt.strftime('%Y-%m-%d')

                # make the call depending on the frequency for how many times to make it
                if target_freq == '1d':
                    params['start_time'] = start_time
                    params['end_time'] = end_time
                    response_json = Helper.makeApiCall(url, headers={}, params=params)
                    if response_json is not None:
                        metric_df = pd.DataFrame(response_json['data'])
                    else:
                        continue
                elif target_freq == '1h':
                    metric_df = pd.DataFrame()
                    time_list = Helper.generateYearlyCalendarYearDateList(start_time, end_time)
                    for j in range(len(time_list)-1):
                        params['start_time'] = time_list[j]
                        params['end_time'] = time_list[j+1]
                        response_json = Helper.makeApiCall(url, headers={}, params=params)
                        if response_json is not None:
                            temp_df = pd.DataFrame(response_json['data'])
                        else: 
                            continue
                        metric_df = pd.concat((metric_df, temp_df))
                    metric_df = metric_df.drop_duplicates(subset=['asset', 'time'])

                # add the results for this asset
                try:
                    asset_df = asset_df.merge(metric_df, on=['asset', 'time'], how='outer', validate='one_to_one')
                except:
                    print(f"for {asset} {metric}, we have no data.")
                    continue

            # add these results to the final df
            df = pd.concat((df, asset_df))

            # periodically output just in case
            df.to_pickle('temp_asset_metrics.pkl')

        return df
    
    @staticmethod
    def cleanUSDPrices(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """ Clean the dataframe with time, asset, and ReferenceRateUSD for USDC and USDT.

        Args:
            df (pd.DataFrame): a DataFrame containing columns for time, asset, and ReferenceRateUSD for USDC and USDT.
            target_freq (str): the target frequency for this study; either 1h or 1d.

        Returns:
            df (pd.DataFrame): a DataFrame with a date column and columns for usd_per_usdc and usd_per_usdt.

        """
        # Clean the columns
        df['date'] = pd.to_datetime(df.time, utc=True).dt.tz_localize(None)
        df['price'] = df.ReferenceRateUSD.astype(float)

        # Sort
        df = df.sort_values(by=['date', 'asset'], ignore_index=True)

        # Ensure all prices are within expected range
        assert 0 == df[(df.price > 2) | (df.price < 0.5)].shape[0]

        # Split into USDC and USDT DataFrames
        usdc_df = df[df.asset == 'usdc'][['date', 'price']]
        usdc_df = usdc_df.rename(columns={'price': 'usd_per_usdc'})
        usdt_df = df[df.asset == 'usdt'][['date', 'price']]
        usdt_df = usdt_df.rename(columns={'price': 'usd_per_usdt'})

        # Ensure that the DataFrames contain consecutive dates
        if target_freq == '1h':
            expected_dates_usdc = pd.Series(pd.date_range(usdc_df['date'].iloc[0], usdc_df['date'].iloc[-1], freq='1H'))
            expected_dates_usdt = pd.Series(pd.date_range(usdt_df['date'].iloc[0], usdt_df['date'].iloc[-1], freq='1H'))
        if target_freq == '1d':
            expected_dates_usdc = pd.Series(pd.date_range(usdc_df['date'].iloc[0], usdc_df['date'].iloc[-1], freq='D'))
            expected_dates_usdt = pd.Series(pd.date_range(usdt_df['date'].iloc[0], usdt_df['date'].iloc[-1], freq='D'))
        assert usdc_df.shape[0] == (expected_dates_usdc.values == pd.to_datetime(usdc_df['date'])).sum()
        assert usdt_df.shape[0] == (expected_dates_usdt.values == pd.to_datetime(usdt_df['date'])).sum()

        # Merge the DataFrames
        df = usdc_df.merge(usdt_df, on='date', how='outer', validate='one_to_one')
        df = df.sort_values(by='date', ignore_index=True)

        return df
        
    @staticmethod
    def pullOHLCV(base_url: str, base_params: Dict[str, str], 
        study_start: str, study_end: str, target_freq: str, markets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a panel DataFrame containing market prices, volumes, and trade counts.

        Args:
            base_url (str): Base URL for the API.
            base_params (Dict[str, str]): Base parameters for the API.
            study_start (str): string time for the start of the study window in format 'YYYY-MM-DD'.
            study_end (str): string time for the end of the study window in format 'YYYY-MM-DD'.
            target_freq (str): the target frequency to study.
            markets_df (pd.DataFrame): A pandas DataFrame containing information about relevant markets.

        Returns:
            A Pandas DataFrame panel of dates and markets with their usd_per_token prices, usd_volume, and trades.
        """
        # Set up object to store data
        df = pd.DataFrame()

        # Form list of markets
        assert markets_df.market.is_unique
        markets_list = list(markets_df.market.values)

        # Build API URL
        api_endpoint = "timeseries/market-candles"
        url = f"{base_url}{api_endpoint}"

        # Define API parameters
        api_params = {
            'frequency': target_freq,
            'page_size': 10000,
            'limit_per_market': 10000
        }

        # Pull all markets
        for i in range(len(markets_list)):
            # update market to pull
            market = markets_list[i]
            params = {**base_params, **api_params, 'markets': market}

            # monitor progress
            print(f"Processing market #{i+1} ({(i+1)/len(markets_list)*100:.2f}%): {market}")

            # determine start and end times for this market
            min_time = markets_df[markets_df.market==market].min_time.values[0]
            max_time = markets_df[markets_df.market==market].max_time.values[0]
            metric_start_dt = datetime.strptime(min_time[:10], '%Y-%m-%d')
            metric_end_dt = datetime.strptime(max_time[:10], '%Y-%m-%d')
            study_start_dt = datetime.strptime(study_start, '%Y-%m-%d')
            study_end_dt = datetime.strptime(study_end, '%Y-%m-%d')
            if study_start_dt >= metric_start_dt:
                start_time = study_start_dt.strftime('%Y-%m-%d')
            else:
                start_time = metric_start_dt.strftime('%Y-%m-%d')
            if study_end_dt <= metric_end_dt:
                end_time = study_end_dt.strftime('%Y-%m-%d')
            else:
                end_time = metric_end_dt.strftime('%Y-%m-%d')

            # make the call depending on the frequency for how many times to make it
            if target_freq == '1d':
                params['start_time'] = start_time
                params['end_time'] = end_time
                response_json = Helper.makeApiCall(url, headers={}, params=params)
                if response_json is not None:
                    result_df = pd.DataFrame(response_json['data'])
                else:
                    continue
            elif target_freq == '1h':
                result_df = pd.DataFrame()
                time_list = Helper.generateYearlyCalendarYearDateList(start_time, end_time)
                for j in range(len(time_list)-1):
                    params['start_time'] = time_list[j]
                    params['end_time'] = time_list[j+1]
                    response_json = Helper.makeApiCall(url, headers={}, params=params)
                    if response_json is not None:
                        temp_df = pd.DataFrame(response_json['data'])
                        result_df = pd.concat((result_df, temp_df))
                    else:
                        continue

            # clean up the results and append
            try:
                result_df = result_df[['market', 'time', 'price_close', 'candle_usd_volume', 'candle_trades_count']]
                result_df = result_df.drop_duplicates(subset=['market', 'time'])
                df = pd.concat((df, result_df))
            except:
                print(f"for {market}, we have no data.")
                continue

        return df
    
    @staticmethod
    def cleanMarketPanels(df: pd.DataFrame, target_freq: str, 
                          markets_df: pd.DataFrame, usd_df: pd.DataFrame, ba_df: pd.DataFrame=None) -> pd.DataFrame:
        """ clean the panel data for markets, closing price, usd volume, and trade counts.

        Args:
            df (pd.DataFrame): DataFrame containing columns for market, time, price_close, 
                               candle_usd_volume, and candle_trades_count.
            target_freq (str): a string for the target frequency for this study.
            markets_df (pd.DataFrame): A pandas DataFrame containing information about relevant markets.
            usd_df (pd.DataFrame): prices for USDC and USDT to calculate exchange rates.
            ba_df (pd.DataFrame): DataFrame containing columns for time, market, ask_price, ask_size, bid_price, bid_size.

        Returns:
            df (pd.DataFrame): a panel dataframe with columns for date, asset, usd_per_token, usd_volume, and trade count.
        """
        # Set indicator if ba_df was passed in
        bid_ask_data = False
        if ba_df is not None:
            bid_ask_data = True
        
        # Confirm no missing obs
        assert 0==df.isnull().sum().sum()

        # Form date column
        df['date'] = pd.to_datetime(df.time, utc=True).dt.tz_localize(None)
        df = df.drop(columns='time', axis=1)

        # Drop duplicates
        df = df.drop_duplicates(subset=['market', 'date'])

        # Cut down to relevant dates
        df = df[df.date <= '2023-01-02']

        # Add market meta data for exchange, base asset, and quote asset
        df = df.merge(markets_df[['market', 'base', 'quote']], on='market', how='inner', validate='many_to_one')
        if bid_ask_data:
            ba_df = ba_df.merge(markets_df[['market', 'base', 'quote']], on='market', how='inner', validate='many_to_one')

        # Merge on USDT and USDC prices
        min_usdt_date = np.min(df[df.quote=='usdt'].date)
        min_usdc_date = np.min(df[df.quote=='usdc'].date)
        assert min_usdt_date >= np.min(usd_df[~usd_df.usd_per_usdt.isnull()].date)
        assert min_usdc_date >= np.min(usd_df[~usd_df.usd_per_usdc.isnull()].date)
        df = df.merge(usd_df, on='date', how='left', validate='many_to_one')
        if bid_ask_data:
            ba_df = ba_df.merge(usd_df, on='date', how='left', validate='many_to_one')
            ba_df = ba_df.drop(columns='market', axis=1)

        # Form price column
        df['price_close'] = df.price_close.astype('float32')
        df.loc[df.quote=='usd', 'usd_per_token_cm'] = df.loc[df.quote=='usd', 'price_close']
        df.loc[df.quote=='usdc', 'usd_per_token_cm'] = df.loc[df.quote=='usdc', 'price_close']*df.loc[df.quote=='usdc', 'usd_per_usdc']
        df.loc[df.quote=='usdt', 'usd_per_token_cm'] = df.loc[df.quote=='usdt', 'price_close']*df.loc[df.quote=='usdt', 'usd_per_usdt']
        assert 0 == df.usd_per_token_cm.isnull().sum()
        df = df.drop(columns=['quote', 'price_close', 'usd_per_usdc', 'usd_per_usdt'], axis=1)
        if bid_ask_data:
            ba_df['ask_price'] = ba_df.ask_price.astype('float32')
            ba_df['bid_price'] = ba_df.bid_price.astype('float32')
            ba_df.loc[ba_df.quote=='usd', 'usd_ask'] = ba_df.loc[ba_df.quote=='usd', 'ask_price']
            ba_df.loc[ba_df.quote=='usdc', 'usd_ask'] = ba_df.loc[ba_df.quote=='usdc', 'ask_price']*ba_df.loc[ba_df.quote=='usdc', 'usd_per_usdc']
            ba_df.loc[ba_df.quote=='usdt', 'usd_ask'] = ba_df.loc[ba_df.quote=='usdt', 'ask_price']*ba_df.loc[ba_df.quote=='usdt', 'usd_per_usdt']
            ba_df.loc[ba_df.quote=='usd', 'usd_bid'] = ba_df.loc[ba_df.quote=='usd', 'bid_price']
            ba_df.loc[ba_df.quote=='usdc', 'usd_bid'] = ba_df.loc[ba_df.quote=='usdc', 'bid_price']*ba_df.loc[ba_df.quote=='usdc', 'usd_per_usdc']
            ba_df.loc[ba_df.quote=='usdt', 'usd_bid'] = ba_df.loc[ba_df.quote=='usdt', 'bid_price']*ba_df.loc[ba_df.quote=='usdt', 'usd_per_usdt']
            assert 0 == ba_df.usd_ask.isnull().sum()
            assert 0 == ba_df.usd_bid.isnull().sum()
            ba_df = ba_df.drop(columns=['ask_price', 'bid_price', 'quote', 'usd_per_usdc', 'usd_per_usdt'], axis=1)

        # Form volume columns
        df['usd_volume_cm'] = df.candle_usd_volume.astype(float)
        df['trades_cm'] = df.candle_trades_count.astype(int)
        assert 0 == df.usd_volume_cm.isnull().sum()
        df = df.drop(columns=['candle_usd_volume', 'candle_trades_count'], axis=1)
        if bid_ask_data:
            ba_df['ask_size'] = ba_df.ask_size.astype('float32')
            ba_df['bid_size'] = ba_df.bid_size.astype('float32')
            ba_df['usd_ask_size'] = ba_df.ask_size*ba_df.usd_ask
            ba_df['usd_bid_size'] = ba_df.bid_size*ba_df.usd_bid
            assert 0 == ba_df.usd_ask_size.isnull().sum()
            assert 0 == ba_df.usd_bid_size.isnull().sum()

        # collapse to the asset date level
        df.loc[df.usd_volume_cm==0, 'usd_volume_cm'] = 1
        grouped = df.groupby(['date', 'base'])
        weighted_avg = grouped.apply(lambda x: (x['usd_per_token_cm'] * x['usd_volume_cm']).sum() / x['usd_volume_cm'].sum())
        total_volume = grouped['usd_volume_cm'].sum()
        total_trades = grouped['trades_cm'].sum()
        df = pd.DataFrame({'usd_per_token_cm': weighted_avg, 
                        'usd_volume_cm': total_volume, 
                        'trades_cm': total_trades}).reset_index()
        df.loc[df.usd_volume_cm==1, 'usd_volume_cm'] = 0
        if bid_ask_data:
            ba_prices_df = ba_df.groupby(['date', 'base'])[['usd_ask', 'usd_bid']].mean()
            ba_vols_df = ba_df.groupby(['date', 'base'])[['usd_ask_size', 'usd_bid_size']].sum()
            ba_df = ba_prices_df.merge(ba_vols_df, on=['date', 'base'], how='outer', validate='one_to_one')
            ba_df = ba_df.reset_index()
            df = df.merge(ba_df, on=['date', 'base'], how='outer', validate='one_to_one')

        # Check for valid ranges and dtypes
        df = df[(df['usd_per_token_cm'] >= 0) & (df['usd_per_token_cm'] < 1e9)]
        df = df[(df['usd_volume_cm'] >= 0) & (df['usd_volume_cm'] < 5e11)]
        df = df[(df['trades_cm'] >= 0) & (df['trades_cm'] < 1e9)]

        # Ensure dtypes are set
        df['usd_per_token_cm'] = df['usd_per_token_cm'].astype('float32')
        df['usd_volume_cm'] = df['usd_volume_cm'].astype('float32')
        df['trades_cm'] = df['trades_cm'].astype('float32')

        # ensure panel is sorted
        df = df.rename(columns={'base': 'asset'})
        df = df.sort_values(by=['date', 'asset'], ignore_index=True)

        # Initial a final dataframe to return
        if bid_ask_data:
            final_df = pd.DataFrame(data={'date': [], 'asset': [], 'usd_per_token_cm': [], 
                                        'usd_volume_cm': [], 'trades_cm': [],
                                        'usd_ask': [], 'usd_bid': [], 'usd_ask_size': [], 'usd_bid_size': []})
        else:
            final_df = pd.DataFrame(data={'date': [], 'asset': [], 'usd_per_token_cm': [], 
                                        'usd_volume_cm': [], 'trades_cm': []})
            
        # Loop over all assets to add any missing days
        assets = list(np.unique(df.asset.values))
        for asset in assets:
            # subset to asset of interest
            asset_df = df[df.asset==asset].copy()

            # determine the date gaps
            date_gaps = []
            dates = asset_df.date.values
            for i in range(1, len(dates)):
                if target_freq == '1d':
                    date_gaps.append(np.timedelta64(dates[i]-dates[i-1], 'D').astype(int))
                elif target_freq == '1h':
                    date_gaps.append(np.timedelta64(dates[i]-dates[i-1], 'h').astype(int))

            # determine new days to add
            if target_freq == '1d':
                indices_to_expand = [i for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32)]
                num_datetime_to_add = [date_gaps[i] for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32)]
                start_datetimes = dates[indices_to_expand]
                new_datetimes = []
                for i in range(len(start_datetimes)):
                    start_datetime = start_datetimes[i]
                    datetime_to_add = num_datetime_to_add[i]
                    for j in range(1, datetime_to_add):
                        new_datetimes.append(start_datetime+np.timedelta64(24*(j), 'h'))
            elif target_freq == '1h':
                indices_to_expand = [i for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32*24)]
                num_datetime_to_add = [date_gaps[i] for i in range(len(date_gaps)) if (date_gaps[i] > 1) & (date_gaps[i] < 32*24)]
                start_datetimes = dates[indices_to_expand]
                new_datetimes = []
                for i in range(len(start_datetimes)):
                    start_datetime = start_datetimes[i]
                    datetime_to_add = num_datetime_to_add[i]
                    for j in range(1, datetime_to_add):
                        new_datetimes.append(start_datetime+np.timedelta64(j, 'h'))
            
            # add the new days to the asset df
            new_asset_df = pd.DataFrame(data={'date': new_datetimes})
            new_asset_df['asset'] = asset
            asset_df = pd.concat((asset_df, new_asset_df))
            asset_df = asset_df.sort_values(by='date', ignore_index=True)

            # forward fill the price columns
            asset_df['usd_per_token_cm'] = asset_df.usd_per_token_cm.ffill()
            if bid_ask_data:
                asset_df['usd_ask'] = asset_df.usd_ask.ffill()
                asset_df['usd_bid'] = asset_df.usd_bid.ffill()

            # replace volume and trades with zeros
            asset_df.loc[asset_df.usd_volume_cm.isnull(), 'usd_volume_cm'] = 0
            asset_df.loc[asset_df.trades_cm.isnull(), 'trades_cm'] = 0
            if bid_ask_data:
                asset_df.loc[asset_df.usd_ask_size.isnull(), 'usd_ask_size'] = 0
                asset_df.loc[asset_df.usd_bid_size.isnull(), 'usd_bid_size'] = 0

            # add data to master df
            final_df = pd.concat((final_df, asset_df))

        # Final clean
        del df
        df = final_df.copy()
        df = df.sort_values(by=['date', 'asset'], ignore_index=True)
        assert not df.duplicated(subset=['date', 'asset']).any()

        return df

    @staticmethod
    def cleanPanel(df: pd.DataFrame) -> pd.DataFrame:
        """ Some light cleaning of the panel as I will do more in the formal cleaning script.
        
        Args: 
            df (pd.DataFrame): panel data of all the asset metrics across asset universe for study period.

        Returns: 
            df (pd.DataFrame): cleaned panel.
        """
        # Form the date column
        df['date'] = pd.to_datetime(df.time, utc=True).dt.tz_localize(None)
        df = df.drop(columns='time', axis=1)

        # Remove bad columns if there are them
        cols_to_drop = [col for col in df.columns if '-status' in col]
        if len(cols_to_drop) > 0:
            df = df.drop(cols_to_drop, axis=1)

        # Cut down to relevant dates
        df = df[df.date <= '2023-01-02']

        # Convert column datatypes
        columns = list(df.columns.values)
        columns.remove('asset')
        columns.remove('date')
        for col in columns:
            df[col] = df[col].astype('float32')

        # Drop duplicates
        df = df.drop_duplicates(subset=['date', 'asset'])

        # Set column order
        df = df[['date', 'asset']+columns]

        # Sort and reset index
        df = df.sort_values(by=['date', 'asset'], ignore_index=True)

        return df
    
    @staticmethod
    def pullExchangeMetrics(base_url: str, base_params: Dict[str, str], target_exchange_metrics: List[str], 
                            study_start: str, study_end: str, target_freq: str) -> pd.DataFrame:
        '''
        Returns a DataFrame containing Coinmetrics exchange metrics.

        Args:
            base_url (str): Base URL for the API.
            base_params (Dict[str, str]): Base parameters for the API.
            target_exchange_metrics (List[str]): List of metrics to pull for exchanges.
            study_start (str): string time for the start of the study window in format 'YYYY-MM-DD'.
            study_end (str): string time for the end of the study window in format 'YYYY-MM-DD'.
            target_freq (str): the target frequency to study.

        Returns:
            pd.DataFrame: A pandas DataFrame containing metrics for the exchanges.
        '''
        # PULL EXCHANGE METRICS METADATA

        # Build target URL and parameters
        api_endpoint = "catalog-all/exchange-metrics"
        url        = f"{base_url}{api_endpoint}"
        params = base_params.copy()

        # Call API and convert to DataFrame
        response_json = Helper.makeApiCall(url, headers={}, params=params)
        ex_metrics_df = pd.DataFrame(response_json['data'])

        # Subset to target exchange metrics
        ex_metrics_df = ex_metrics_df[ex_metrics_df.metric.isin(target_exchange_metrics)]

        # PULL EXCHANGE METRICS

        # Update url and params for exchange metric pull
        api_endpoint = "timeseries/exchange-metrics"
        url        = f"{base_url}{api_endpoint}"
        api_params = {'page_size': 10000,
                    'end_inclusive': False,
                    'limit_per_exchange': 10000,
                    'frequency': target_freq}
        params = {**base_params, **api_params}

        # Initialize a DataFrame for the results
        df = pd.DataFrame(data={'exchange': [], 'time': []})

        # Loop over the metrics to pull
        for metric in target_exchange_metrics:
            # monitor progress
            print(f"Working on exchange metric {metric}.")

            # Update parameters
            params['metrics'] = metric

            # pull exchanges for this metric
            exchanges = ex_metrics_df[ex_metrics_df.metric==metric].frequencies.values[0][0]['exchanges']

            # Initialize DataFrame for the metric
            metric_df = pd.DataFrame()

            # Loop over the exchanges for this metric
            for exchange in exchanges:
                # Update parameters
                params['exchanges'] = exchange

                # Form list of times to pull
                time_list = Helper.generateYearlyCalendarYearDateList(study_start, study_end)

                # Pull for all time periods
                for j in range(len(time_list)-1):
                    # Set time period to pull a calendar year at a time
                    params['start_time'] = time_list[j]
                    params['end_time'] = time_list[j+1]

                    # Make the call and extract if there is data
                    response_json = Helper.makeApiCall(url, headers={}, params=params)
                    if response_json is not None:
                        temp_df = pd.DataFrame(response_json['data'])
                        metric_df = pd.concat([metric_df, temp_df])
                    else: 
                        continue
        
            # Drop duplicates
            metric_df = metric_df.drop_duplicates(subset=['exchange', 'time'])
                    
            # Merge these exchange metrics to the master DataFrame
            df = df.merge(metric_df, on=['exchange', 'time'], how='outer', validate='one_to_one')
        
        return df

    @staticmethod
    def cleanExchangeMetrics(df: pd.DataFrame, target_exchange_metrics: List[str], target_us_exchanges: List[str]) -> pd.DataFrame:
        ''' Clean the DataFrame containing metrics at the datetime exchange level.
        
        Args:
            df (pd.DataFrame): raw data of exchange, time, and metrics.
            target_exchange_metrics (List[str]): List of metrics to pull for exchanges.
            target_us_exchanges (List[str]): A list of strings with the target exchanges for this study.
        
        Returns:
            df (pd.DataFrame): clean data at day level of exchange metrics and US exchange metrics.
        '''
        # Confirm data type
        assert str == type(df.exchange.values[0])

        # Form the date column
        df['date'] = pd.to_datetime(df.time, utc=True).dt.tz_localize(None)
        df = df.drop(columns='time', axis=1)

        # Convert dtypes
        cols = list(df.columns.values)
        cols.remove('exchange')
        cols.remove('date')
        for col in cols:
            df[col] = df[col].astype('float32')

        # Collapse to datetime level
        all_df = df.groupby('date')[target_exchange_metrics].sum()
        us_df  = df[df.exchange.isin(target_us_exchanges)].groupby('date')[target_exchange_metrics].sum()

        # Rename columns
        for col in cols:
            all_df = all_df.rename(columns={col: f"ex_{col}"})
            us_df  = us_df.rename(columns={col: f"us_ex_{col}"})

        # Merge together
        df = all_df.merge(us_df, on=['date'], how='outer', validate='one_to_one')
        df = df.reset_index()

        # Assert consecutive hours for all metrics
        total_hours_expected = 1+np.timedelta64(np.max(df.date)-np.min(df.date), 'h').astype(int)
        assert df.shape[0] == total_hours_expected

        # Confirm no missing obs
        assert 0 == df.isnull().sum().sum()

        # Confirm no duplicates
        assert df.date.is_unique

        return df.sort_values(by='date', ignore_index=True)

    @staticmethod
    def pullBidAsk(base_url: str, base_params: Dict[str, str], 
                   study_start: str, study_end: str, markets_df: pd.DataFrame) -> pd.DataFrame:
        """ Returns a panel DataFrame containing time, market, bid price, bid size, ask price, and ask size.

        Args:
            base_url (str): Base URL for the API.
            base_params (Dict[str, str]): Base parameters for the API.
            study_start (str): string time for the start of the study window in format 'YYYY-MM-DD'.
            study_end (str): string time for the end of the study window in format 'YYYY-MM-DD'.
            markets_df (pd.DataFrame): A pandas DataFrame containing information about relevant markets.

        Returns:
            A pandas DataFrame panel of dates and markets with their ask_price, ask_size, bid_price, bid_size.
        
        """
        # Initialize a DataFrame for the results
        df = pd.DataFrame()

        # Form list of markets
        assert markets_df.market.is_unique
        markets_list = list(markets_df.market.values)

        # Build API URL
        api_endpoint = "timeseries/market-quotes"
        url = f"{base_url}{api_endpoint}"

        # Define API parameters
        api_params = {
            'page_size': 10000,
            'limit_per_market': 10000,
            'end_inclusive': False
        }

        # Loop over the markets to pull bid ask data for
        for i in range(len(markets_list)):
            # update market to pull
            market = markets_list[i]
            params = {**base_params, **api_params, 'markets': market}

            # monitor progress
            print(f"Pulling bid ask data for market #{i+1} ({(i+1)/len(markets_list)*100:.2f}%): {market}")

            # determine start and end times for this market
            min_time = markets_df[markets_df.market==market].min_time.values[0]
            max_time = markets_df[markets_df.market==market].max_time.values[0]
            metric_start_dt = datetime.strptime(min_time[:10], '%Y-%m-%d')
            metric_end_dt = datetime.strptime(max_time[:10], '%Y-%m-%d')
            study_start_dt = datetime.strptime(study_start, '%Y-%m-%d')
            study_end_dt = datetime.strptime(study_end, '%Y-%m-%d')
            if study_start_dt >= metric_start_dt:
                start_time = study_start_dt.strftime('%Y-%m-%d')
            else:
                start_time = metric_start_dt.strftime('%Y-%m-%d')
            if study_end_dt <= metric_end_dt:
                end_time = study_end_dt.strftime('%Y-%m-%d')
            else:
                end_time = metric_end_dt.strftime('%Y-%m-%d')
            
            # Make the API call
            time_list = Helper.generateDailyDateList(start_time, end_time)
            for j in range(len(time_list)-1):
                params['start_time'] = time_list[j]
                params['end_time'] = time_list[j+1]
                response_json = Helper.makeApiCall(url, headers={}, params=params)
                time.sleep(0.1)
                if type(response_json) == dict:
                    if 'data' in response_json.keys():
                        if len(response_json['data']) > 0:
                            # convert to DataFrame
                            temp_df = pd.DataFrame(response_json['data'])

                            # convert time column to a numpy.datetime64 type
                            temp_df['date'] =  pd.to_datetime(temp_df.time, utc=True).dt.tz_localize(None)
                            temp_df = temp_df.drop(columns='time', axis=1)

                            # resample the data to the hourly level keeping the last observation in each hour
                            temp_df = temp_df.sort_values(by='date')
                            temp_df['date'] = temp_df.date.dt.ceil('H')
                            temp_df = temp_df.drop_duplicates(subset='date', keep='last')

                            # subset and order columns and add to master DataFrame
                            temp_df = temp_df[['date', 'market', 'ask_price', 'ask_size', 'bid_price', 'bid_size']]
                            df = pd.concat((df, temp_df))

        
        return df.sort_values(by='date', ignore_index=True)