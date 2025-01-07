"""The Kupo Python client."""

import os
from collections.abc import Callable
from collections.abc import Iterator
from enum import Enum
from functools import partial
from functools import wraps
from typing import Any
from typing import Self

import aiohttp as aio  # type: ignore
import requests
from dotenv import load_dotenv  # type: ignore
from pycardano import Address  # type: ignore
from pycardano import AssetName  # type: ignore
from pycardano import ScriptHash  # type: ignore
from pycardano import TransactionId  # type: ignore
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel
from pydantic import field_validator
from pydantic import model_validator

from kupo.models import Datum
from kupo.models import Match
from kupo.models import Metadata
from kupo.models import Pattern
from kupo.models import Point
from kupo.models import Script

load_dotenv()

KUPO_BASE_URL = os.environ.get("KUPO_BASE_URL", None)
KUPO_PORT = os.environ.get("KUPO_PORT", None)


class Order(Enum):
    """Order of results by time."""

    DESC = "most_recent_first"
    ASC = "least_recent_first"


class Limit(Enum):
    """Order of results by time."""

    SAFE = "within_safe_zone"
    UNSAFE = "unsafe_allow_beyond_safe_zone"


class QueryBase(BaseModel):
    """Query paramete base model, includes utility functions."""

    def params(self) -> dict[str, Any]:
        """Paramters for the query."""
        return self.model_dump(mode="json", exclude_none=True)


class BaseList(RootModel):
    """Base class for array responses."""

    def __getitem__(self, index: int) -> RootModel | str:
        """Get an item from the list."""
        return self.root[index]

    def __len__(self) -> int:
        """Get the length of the list."""
        return len(self.root)

    def __iter__(self) -> Iterator[RootModel | str]:
        """Iterate over the list."""
        return iter(self.root)


class MatchQuery(QueryBase):
    """Validator for the match query parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    resolve_hashes: bool = Field(False, exclude=True)
    spent: bool = Field(False, exclude=True)
    unspent: bool = Field(False, exclude=True)
    order: Order = Order.DESC
    created_after: int | None = None
    spent_after: int | None = None
    created_before: int | None = None
    spent_before: int | None = None
    policy_id: str | ScriptHash | None = None
    asset_name: str | AssetName | None = None
    transaction_id: str | TransactionId | None = None
    output_index: int | None = None

    def flags(self) -> list[str]:
        """Get flags, which are empty parameters in the query string."""
        flags = []
        if self.spent:
            flags.append("spent")
        elif self.unspent:
            flags.append("unspent")

        if self.resolve_hashes:
            flags.append("resolve_hashes")

        return flags

    @field_validator(
        "spent_after",
        "spent_before",
        "created_after",
        "created_before",
        mode="before",
    )
    @classmethod
    def validate_int(cls, v: float | str) -> int | None:
        """Convert timestamps to integers."""
        return None if v is None else int(v)

    @field_validator(
        "policy_id",
        "asset_name",
        "transaction_id",
        mode="before",
    )
    @classmethod
    def validate_pycardano(
        cls,
        v: str | ScriptHash | AssetName | TransactionId | None,
    ) -> int | str | None:
        """Convert timestamps to integers."""
        if isinstance(v, ScriptHash | AssetName | TransactionId):
            v = v.payload.hex()
        return v

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        """Run validation on match parameters to comply with API requirements."""
        if self.spent and self.unspent:
            raise ValueError("Cannot have both spent and unspent in parameters")

        if self.spent_after is not None and self.created_after is not None:
            raise ValueError(
                "Cannot have both spent_after and created_after in parameters",
            )

        if self.spent_before is not None and self.created_before is not None:
            raise ValueError(
                "Cannot have both spent_before and created_before in parameters",
            )

        if (
            self.spent_after is not None
            and self.spent_before is not None
            and self.spent_after > self.spent_before
        ):
            raise ValueError("spent_after must be less than spent_before")

        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise ValueError("created_after must be less than created_before")

        if self.unspent and (
            self.spent_after is not None or self.spent_before is not None
        ):
            raise ValueError("Cannot have spent_after or spent_before with unspent")

        if (
            self.spent
            and self.spent_after is not None
            and self.spent_before is not None
        ):
            raise ValueError(
                "spent is redundant when spent_after and spent_before are set",
            )

        return self


class MatchResponse(BaseList):
    """The response model for endpoints that return matches."""

    root: list[Match]


class PatternResponse(BaseList):
    """The response model for endpoints that return patterns."""

    root: list[Pattern]


def rest_wrapper(
    func: Callable | None = None,
    param_model: type[BaseModel] | type[RootModel] | None = None,
    is_async: bool = False,
) -> Callable:
    """Autoparses input arguments to the given model."""
    if param_model is None:
        raise ValueError("Both param and response models must be supplied")

    if func is None:
        return partial(rest_wrapper, param_model=param_model, is_async=is_async)

    if is_async:

        @wraps(func)
        async def async_wrapped(*args, **kwargs):  # noqa
            if "model" not in kwargs or kwargs["model"] is None:
                parsed = param_model.model_validate(kwargs)
                kwargs["model"] = parsed
            return await func(*args, **kwargs)

        return async_wrapped

    @wraps(func)
    def wrapped(*args, **kwargs):  # noqa
        if "model" not in kwargs or kwargs["model"] is None:
            parsed = param_model.model_validate(kwargs)
            kwargs["model"] = parsed
        return func(*args, **kwargs)

    return wrapped


class KupoClient:
    """A Python client for the Kupo REST API."""

    def __init__(self, base_url: str | None = None, port: int | None = None) -> None:
        """The Python Kupo API client.

        Args:
            base_url: If not supplied, tries to read from the environment.
                Defaults to None.
            port: If not supplied, tries to read from the environment. Defaults to None.
        """
        if base_url is None:
            base_url = KUPO_BASE_URL

        if port is None and KUPO_PORT is not None:
            port = int(KUPO_PORT)

        if base_url is None:
            raise ValueError("base_url must be supplied")

        if port is None:
            raise ValueError("port must be supplied")

        self._base_url = base_url
        self._port = port

    @property
    def base_url(self) -> str:
        """The base URL for the client."""
        return f"{self._base_url}:{self._port}"

    async def _get_async(
        self,
        path: str,
        params: dict[str, Any] = {},
        flags: list[str] = [],
        timeout: int = 300,
    ) -> dict | list:
        path = self.base_url + path
        if len(flags) > 0:
            path += "?" + "&".join(flags)
        async with aio.ClientSession() as session:
            async with session.get(
                path,
                params=params,
                timeout=aio.ClientTimeout(total=timeout),
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _get(
        self,
        path: str,
        params: dict[str, Any] = {},
        flags: list[str] = [],
        timeout: int = 300,
    ) -> dict | list:
        with requests.session() as session:
            path = self.base_url + path
            if len(flags) > 0:
                path += "?" + "&".join(flags)
            response = session.get(path, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()

    async def _put_async(
        self,
        path: str,
        body: dict[str, Any] = {},
        timeout: int = 300,
    ) -> dict | list:
        async with aio.ClientSession() as session:
            path = self.base_url + path
            async with session.put(
                path,
                json=body,
                timeout=aio.ClientTimeout(total=timeout),
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _put(
        self,
        path: str,
        body: dict[str, Any] = {},
        timeout: int = 300,
    ) -> dict | list:
        with requests.session() as session:
            path = self.base_url + path
            response = session.put(path, json=body, timeout=timeout)
            response.raise_for_status()
            return response.json()

    @rest_wrapper(param_model=MatchQuery)
    def get_all_matches(
        self,
        resolve_hashes: bool = False,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | ScriptHash | None = None,
        asset_name: str | AssetName | None = None,
        transaction_id: str | TransactionId | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            resolve_hashes: Resolve hashes to human readable strings. Defaults to False.
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after slot number. Defaults to None.
            spent_after: UTxO spent after slot number. Defaults to None.
            created_before: UTxO created before slot number. Defaults to None.
            spent_before: UTxO spent before slot number. Defaults to None.
            policy_id: UTxO contains token with policy id. Defaults to None.
            asset_name: UTxO contains token with policy name. Defaults to None.
            transaction_id: UTxOs with transaction id. Defaults to None.
            output_index: Index of output. Defaults to None.
            timeout: The timeout in seconds. Defaults to 300.
            model: The MatchQuery model. Will use if supplied. Defaults to None.

        Returns:
            _description_
        """
        if model is None:
            raise ValueError("model must be None")

        response = self._get(
            "/matches",
            params=model.params(),
            flags=model.flags(),
            timeout=timeout,
        )

        return MatchResponse.model_validate(response)

    @rest_wrapper(param_model=MatchQuery, is_async=True)
    async def get_all_matches_async(
        self,
        resolve_hashes: bool = False,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | ScriptHash | None = None,
        asset_name: str | AssetName | None = None,
        transaction_id: str | TransactionId | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            resolve_hashes: Resolve hashes to human readable strings. Defaults to False.
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after slot number. Defaults to None.
            spent_after: UTxO spent after slot number. Defaults to None.
            created_before: UTxO created before slot number. Defaults to None.
            spent_before: UTxO spent before slot number. Defaults to None.
            policy_id: UTxO contains token with policy id. Defaults to None.
            asset_name: UTxO contains token with policy name. Defaults to None.
            transaction_id: UTxOs with transaction id. Defaults to None.
            output_index: Index of output. Defaults to None.
            timeout: The timeout in seconds. Defaults to 300.
            model: The MatchQuery model. Will use if supplied. Defaults to None.

        Returns:
            _description_
        """
        if model is None:
            raise ValueError("model must be None")

        response = await self._get_async(
            "/matches",
            params=model.params(),
            flags=model.flags(),
            timeout=timeout,
        )

        return MatchResponse.model_validate(response)

    @rest_wrapper(param_model=MatchQuery)
    def get_matches(
        self,
        pattern: str | Address,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | ScriptHash | None = None,
        asset_name: str | AssetName | None = None,
        transaction_id: str | TransactionId | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            pattern: The pattern to match.
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after slot number. Defaults to None.
            spent_after: UTxO spent after slot number. Defaults to None.
            created_before: UTxO created before slot number. Defaults to None.
            spent_before: UTxO spent before slot number. Defaults to None.
            policy_id: UTxO contains token with policy id. Defaults to None.
            asset_name: UTxO contains token with policy name. Defaults to None.
            transaction_id: UTxOs with transaction id. Defaults to None.
            output_index: Index of output. Defaults to None.
            timeout: The timeout in seconds. Defaults to 300.
            model: The MatchQuery model. Will use if supplied. Defaults to None.

        Returns:
            The MatchResponse model.
        """
        if model is None:
            raise ValueError("model must be None")

        response = self._get(
            f"/matches/{pattern}",
            params=model.params(),
            flags=model.flags(),
            timeout=timeout,
        )

        return MatchResponse.model_validate(response)

    @rest_wrapper(param_model=MatchQuery, is_async=True)
    async def get_matches_async(
        self,
        pattern: str | Address,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | ScriptHash | None = None,
        asset_name: str | AssetName | None = None,
        transaction_id: str | TransactionId | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            pattern: The pattern to match.
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after slot number. Defaults to None.
            spent_after: UTxO spent after slot number. Defaults to None.
            created_before: UTxO created before slot number. Defaults to None.
            spent_before: UTxO spent before slot number. Defaults to None.
            policy_id: UTxO contains token with policy id. Defaults to None.
            asset_name: UTxO contains token with policy name. Defaults to None.
            transaction_id: UTxOs with transaction id. Defaults to None.
            output_index: Index of output. Defaults to None.
            timeout: The timeout in seconds. Defaults to 300.
            model: The MatchQuery model. Will use if supplied. Defaults to None.

        Returns:
            The MatchResponse model.
        """
        if model is None:
            raise ValueError("model must be None")

        response = await self._get_async(
            f"/matches/{pattern}",
            params=model.params(),
            flags=model.flags(),
            timeout=timeout,
        )

        return MatchResponse.model_validate(response)

    def get_patterns(self, timeout: int = 300) -> PatternResponse:
        """Get all patterns used for matching.

        Args:
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The list of matching patterns.
        """
        results = self._get("/patterns", timeout=timeout)

        return PatternResponse.model_validate(results)

    async def get_patterns_async(self, timeout: int = 300) -> PatternResponse:
        """Get all patterns used for matching.

        Args:
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The list of matching patterns.
        """
        results = await self._get_async("/patterns", timeout=timeout)

        return PatternResponse.model_validate(results)

    def get_pattern(self, pattern: str, timeout: int = 300) -> PatternResponse:
        """Get all patterns that match a given pattern (if possible).

        Args:
            pattern: The pattern to match.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The list of matching patterns.
        """
        results = self._get(f"/patterns/{pattern}", timeout=timeout)

        return PatternResponse.model_validate(results)

    async def get_pattern_async(
        self,
        pattern: str,
        timeout: int = 300,
    ) -> PatternResponse:
        """Get all patterns that match a given pattern (if possible).

        Args:
            pattern: The pattern to match.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The list of matching patterns.
        """
        results = await self._get_async(f"/patterns/{pattern}", timeout=timeout)

        return PatternResponse.model_validate(results)

    def get_script_by_hash(self, script_hash: str, timeout: int = 300) -> Script:
        """Get a script by its hash.

        Args:
            script_hash: The hash of the script to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The script matching the hash.
        """
        results = self._get(f"/scripts/{script_hash}", timeout=timeout)

        return Script.model_validate(results)

    async def get_script_by_hash_async(
        self,
        script_hash: str,
        timeout: int = 300,
    ) -> Script:
        """Get a script by its hash.

        Args:
            script_hash: The hash of the script to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The script matching the hash.
        """
        results = await self._get_async(f"/scripts/{script_hash}", timeout=timeout)

        return Script.model_validate(results)

    def get_datum_by_hash(self, datum_hash: str, timeout: int = 300) -> Datum:
        """Get a datum by its hash.

        Args:
            datum_hash: The hash of the datum to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The datum matching the hash.
        """
        results = self._get(f"/datums/{datum_hash}", timeout=timeout)

        return Datum.model_validate(results)

    async def get_datum_by_hash_async(
        self,
        datum_hash: str,
        timeout: int = 300,
    ) -> Datum:
        """Get a datum by its hash.

        Args:
            datum_hash: The hash of the datum to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The datum matching the hash.
        """
        results = await self._get_async(f"/datums/{datum_hash}", timeout=timeout)

        return Datum.model_validate(results)

    def get_metadata_by_tx(self, tx_hash: str, timeout: int = 300) -> Metadata:
        """Get a metadata by transaction hash.

        Args:
            tx_hash: The hash of the metadata to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The metadata matching the hash.
        """
        results = self._get(f"/metadata/{tx_hash}", timeout=timeout)

        return Metadata.model_validate(results)

    async def get_metadata_by_tx_async(
        self,
        tx_hash: str,
        timeout: int = 300,
    ) -> Metadata:
        """Get a datum by its hash.

        Args:
            tx_hash: The hash of the metadata to fetch.
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The metadata matching the hash.
        """
        results = await self._get_async(f"/metadata/{tx_hash}", timeout=timeout)

        return Metadata.model_validate(results)

    def add_pattern(
        self,
        pattern: str,
        rollback_to: int | dict | Point,
        limit: Limit = Limit.SAFE,
        timeout: int = 300,
    ) -> PatternResponse:
        """Add a pattern to the database.

        Args:
            pattern: The capture pattern.
            rollback_to: The point to rollback to. Must be either a slot number (int),
                a Point object, or a dictionary with keys `slot_no` (required) and
                `header_hash` (optional).
            limit: Use safe or unsafe boundaries as defined in the docs.
                Defaults to Limit.SAFE.
            timeout: Timeout for the request. Defaults to 300.

        Returns:
            A PatternResponse object.
        """
        if isinstance(rollback_to, Point):
            rollback_to = rollback_to.model_dump(mode="json")
        elif isinstance(rollback_to, int):
            rollback_to = {"slot_no": rollback_to}

        if "slot_no" not in rollback_to:
            raise ValueError("rollback_to must contain slot_no")

        results = self._put(
            f"/patterns/{pattern}",
            body={"rollback_to": rollback_to, "limit": limit.value},
            timeout=timeout,
        )

        return PatternResponse.model_validate(results)

    async def add_pattern_async(
        self,
        pattern: str,
        rollback_to: int | dict | Point,
        limit: Limit = Limit.SAFE,
        timeout: int = 300,
    ) -> PatternResponse:
        """Add a pattern to the database.

        Args:
            pattern: The capture pattern.
            rollback_to: The point to rollback to. Must be either a slot number (int),
                a Point object, or a dictionary with keys `slot_no` (required) and
                `header_hash` (optional).
            limit: Use safe or unsafe boundaries as defined in the docs.
                Defaults to Limit.SAFE.
            timeout: Timeout for the request. Defaults to 300.

        Returns:
            A PatternResponse object.
        """
        if isinstance(rollback_to, Point):
            rollback_to = rollback_to.model_dump(mode="json")
        elif isinstance(rollback_to, int):
            rollback_to = {"slot_no": rollback_to}

        if "slot_no" not in rollback_to:
            raise ValueError("rollback_to must contain slot_no")

        results = await self._put_async(
            f"/patterns/{pattern}",
            body={"rollback_to": rollback_to, "limit": limit.value},
            timeout=timeout,
        )

        return PatternResponse.model_validate(results)
