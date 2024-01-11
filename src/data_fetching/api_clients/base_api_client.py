from typing import Any, Dict, Optional

import requests


class BaseAPI:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Private method for GET requests."""
        try:
            response = requests.get(self.base_url + endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
