import requests
import time
import logging
logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional
import random

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
            except requests.exceptions.RequestException as e:
                logger.warning(f'The API call failed with error: {str(e)}')
                if attempt == retries - 1:
                    logger.error(f'The API call failed after {retries} attempts.')
                    return None
                else:
                    sleep_time = (4 ** attempt) * (0.5 + random.uniform(0, 1))
                    logger.warning(f'Retrying after {sleep_time:.2f} seconds.')
                    time.sleep(sleep_time)
