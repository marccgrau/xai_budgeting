from typing import Any, Dict, Optional

from src.data_fetching.api_clients.base_api_client import BaseAPI


class SGAPI(BaseAPI):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)

    def get_catalog_datasets(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Query catalog datasets."""
        return self._get("/catalog/datasets", params=params)

    def get_catalog_exports(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """List available export formats."""
        return self._get("/catalog/exports", params=params)

    def export_catalog(
        self, format: str = "csv", params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Export a catalog in a specified format. Default is CSV."""
        return self._get(f"/catalog/exports/{format}", params=params)

    def list_facet_values(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """List values of different facets."""
        return self._get("/catalog/facets", params=params)

    def get_dataset_info(
        self, dataset_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Retrieve information about a specific dataset."""
        return self._get(f"/catalog/datasets/{dataset_id}", params=params)

    def query_dataset_records(
        self, dataset_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query records within a specific dataset."""
        return self._get(f"/catalog/datasets/{dataset_id}/records", params=params)

    def list_dataset_exports(
        self, dataset_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """List export formats for a specific dataset."""
        return self._get(f"/catalog/datasets/{dataset_id}/exports", params=params)

    def export_dataset(
        self,
        dataset_id: str,
        format: str = "csv",
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Export a dataset in a specified format. Default is CSV."""
        return self._get(
            f"/catalog/datasets/{dataset_id}/exports/{format}", params=params
        )

    def list_dataset_facets(
        self, dataset_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """List facets for a specific dataset."""
        return self._get(f"/catalog/datasets/{dataset_id}/facets", params=params)

    def list_dataset_attachments(
        self, dataset_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Retrieve a list of attachments associated with a dataset."""
        return self._get(f"/catalog/datasets/{dataset_id}/attachments", params=params)

    def read_dataset_record(
        self, dataset_id: str, record_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Retrieve a specific record from a dataset."""
        return self._get(
            f"/catalog/datasets/{dataset_id}/records/{record_id}", params=params
        )
