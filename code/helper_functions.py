import requests
import time
import logging
logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional, List
import random
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import chain

class Helper:
    """ Class of helper functions specific to crypto asset pricing project. """

    @staticmethod
    def makeApiCall(url: str, headers: Optional[Dict[str, str]], params: Optional[Dict[str, str]] = None, 
                    retries: int = 4, timeout: int = 5) -> Optional[Dict[str, Any]]:
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
                response = requests.get(url, headers=headers, params=params, timeout=5)
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
            df: Pandas DataFrame of date, asset_id, usd_per_token, usd_volume_per_24h, 
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
        df['usd_volume_per_24h_coinapi'] = df.volume_traded*df.usd_per_token_coinapi
        assert 0 == df.usd_volume_per_24h_coinapi.isnull().sum()

        # collapse to the asset date level
        df.loc[df.usd_volume_per_24h_coinapi==0, 'usd_volume_per_24h_coinapi'] = 1
        grouped = df.groupby(['date', 'asset_id'])
        weighted_avg = grouped.apply(lambda x: (x['usd_per_token_coinapi'] * x['usd_volume_per_24h_coinapi']).sum() / x['usd_volume_per_24h_coinapi'].sum())
        total_volume = grouped['usd_volume_per_24h_coinapi'].sum()
        total_trades = grouped['trades_count'].sum()
        df = pd.DataFrame({'usd_per_token_coinapi': weighted_avg, 
                            'usd_volume_per_24h_coinapi': total_volume, 
                            'trades_count': total_trades}).reset_index()
        df.loc[df.usd_volume_per_24h_coinapi==1, 'usd_volume_per_24h_coinapi'] = 0

        # check for valid ranges and dtypes
        assert 0 == df.usd_per_token_coinapi.isnull().sum()
        assert 0 == df.usd_volume_per_24h_coinapi.isnull().sum()
        df = df[(df['usd_per_token_coinapi'] >= 0) & (df['usd_per_token_coinapi'] < 1e9)]
        df = df[(df['usd_volume_per_24h_coinapi'] >= 0) & (df['usd_volume_per_24h_coinapi'] < 5e11)]
        df = df[(df['trades_count'] >= 0) & (df['trades_count'] < 1e9)]

        # ensure dtypes are set
        df['usd_per_token_coinapi'] = df['usd_per_token_coinapi'].astype('float32')
        df['usd_volume_per_24h_coinapi'] = df['usd_volume_per_24h_coinapi'].astype('float32')
        df['trades_count'] = df['trades_count'].astype('float32')

        # ensure panel is sorted
        df = df.sort_values(by=['date', 'asset_id'], ignore_index=True)
        df['date'] = pd.to_datetime(df.date)


        # initialize a new df
        final_df = pd.DataFrame(data={'date': [], 'asset_id': [], 'usd_per_token_coinapi': [], 'usd_volume_per_24h_coinapi': [], 'trades_count': []})

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
            asset_df.loc[asset_df.usd_volume_per_24h_coinapi.isnull(), 'usd_volume_per_24h_coinapi'] = 0
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