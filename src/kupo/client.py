import os
import time
from enum import Enum
from functools import partial
from functools import wraps
from typing import Any, Callable, Self

import aiohttp as aio
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import RootModel

from kupo.models import Datum
from kupo.models import Match
from kupo.models import Metadata
from kupo.models import Pattern
from kupo.models import Script

load_dotenv()

KUPO_BASE_URL = os.environ.get("KUPO_BASE_URL", None)
KUPO_PORT = os.environ.get("KUPO_PORT", None)


class Order(Enum):
    DESC = "most_recent_first"
    ASC = "least_recent_first"


class QueryBase(BaseModel):

    def params(self) -> dict[str, Any]:

        return self.model_dump(mode="json", exclude_none=True)


class BaseArray(RootModel):

    def __getitem__(self, index: int) -> Match:
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self):
        return self.root


class MatchQuery(QueryBase):
    spent: bool = Field(False, exclude=True)
    unspent: bool = Field(False, exclude=True)
    order: Order = Order.DESC
    created_after: int | None = None
    spent_after: int | None = None
    created_before: int | None = None
    spent_before: int | None = None
    policy_id: str | None = None
    asset_name: str | None = None
    transaction_id: str | None = None
    output_index: int | None = None

    def flags(self) -> list[str]:
        flags = []
        if self.spent:
            flags.append("spent")
        elif self.unspent:
            flags.append("unspent")

        return flags

    @field_validator(
        "spent_after", "spent_before", "created_after", "created_before", mode="before"
    )
    def validate_int(cls, v: Any) -> int | None:

        return None if v is None else int(v)

    @model_validator(mode="after")
    def validate_params(self) -> Self:

        if self.spent and self.unspent:
            raise ValueError("Cannot have both spent and unspent in parameters")

        if self.spent_after is not None and self.created_after is not None:
            raise ValueError(
                "Cannot have both spent_after and created_after in parameters"
            )

        if self.spent_before is not None and self.created_before is not None:
            raise ValueError(
                "Cannot have both spent_before and created_before in parameters"
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
                "spent is redundant when spent_after and spent_before are set"
            )

        return self


class MatchResponse(BaseArray):
    root: list[Match]


class PatternResponse(BaseArray):
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
        async def wrapped(*args, **kwargs):
            if "model" not in kwargs or kwargs["model"] is None:
                parsed = param_model.model_validate(kwargs)
                kwargs["model"] = parsed
            return await func(*args, **kwargs)

        return wrapped

    else:

        @wraps(func)
        def wrapped(*args, **kwargs):
            if "model" not in kwargs or kwargs["model"] is None:
                parsed = param_model.model_validate(kwargs)
                kwargs["model"] = parsed
            return func(*args, **kwargs)

        return wrapped


class KupoClient:
    def __init__(self, base_url: str | None = None, port: int | None = None):
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
                path, params=params, timeout=aio.ClientTimeout(total=timeout)
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

    @rest_wrapper(param_model=MatchQuery)
    def get_all_matches(
        self,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | None = None,
        asset_name: str | None = None,
        transaction_id: str | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after. Defaults to None.
            spent_after: UTxO spent after. Defaults to None.
            created_before: UTxO created before. Defaults to None.
            spent_before: UTxO spent before. Defaults to None.
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
            "/matches", params=model.params(), flags=model.flags(), timeout=timeout
        )

        return MatchResponse.model_validate(response)

    @rest_wrapper(param_model=MatchQuery, is_async=True)
    async def get_all_matches_async(
        self,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | None = None,
        asset_name: str | None = None,
        transaction_id: str | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get all matches.

        Args:
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after. Defaults to None.
            spent_after: UTxO spent after. Defaults to None.
            created_before: UTxO created before. Defaults to None.
            spent_before: UTxO spent before. Defaults to None.
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
            "/matches", params=model.params(), flags=model.flags(), timeout=timeout
        )

        return MatchResponse.model_validate(response)

    @rest_wrapper(param_model=MatchQuery)
    def get_matches(
        self,
        pattern: str,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | None = None,
        asset_name: str | None = None,
        transaction_id: str | None = None,
        output_index: int | None = None,
        timeout: int = 300,
        model: MatchQuery | None = None,
    ) -> MatchResponse:
        """Get matches for a pattern.

        Args:
            pattern: The pattern to match.
            spent: Only return spent transactions. Defaults to False.
            unspent: Only return unspect transactions. Defaults to False.
            order: Must be Orders.ASC or Orders.DESC. Defaults to Order.DESC.
            created_after: UTxO created after. Defaults to None.
            spent_after: UTxO spent after. Defaults to None.
            created_before: UTxO created before. Defaults to None.
            spent_before: UTxO spent before. Defaults to None.
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
        pattern: str,
        spent: bool = False,
        unspent: bool = False,
        order: Order = Order.DESC,
        created_after: int | str | None = None,
        spent_after: int | str | None = None,
        created_before: int | str | None = None,
        spent_before: int | str | None = None,
        policy_id: str | None = None,
        asset_name: str | None = None,
        transaction_id: str | None = None,
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
            created_after: UTxO created after. Defaults to None.
            spent_after: UTxO spent after. Defaults to None.
            created_before: UTxO created before. Defaults to None.
            spent_before: UTxO spent before. Defaults to None.
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

        results = self._get(f"/patterns", timeout=timeout)

        return PatternResponse.model_validate(results)

    async def get_patterns_async(self, timeout: int = 300) -> PatternResponse:
        """Get all patterns used for matching.

        Args:
            timeout: The timeout in seconds. Defaults to 300.

        Returns:
            The list of matching patterns.
        """

        results = await self._get_async(f"/patterns", timeout=timeout)

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
        self, pattern: str, timeout: int = 300
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
        self, script_hash: str, timeout: int = 300
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
        self, datum_hash: str, timeout: int = 300
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

    async def get_script_by_hash_async(
        self, script_hash: str, timeout: int = 300
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
        self, tx_hash: str, timeout: int = 300
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
