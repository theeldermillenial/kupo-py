import json
import os

import pytest
from kupo.dbsync import get_matches as get_matches_dbsync


def test_compare_matches(
    kupo_client,
    pattern,
    spent,
    unspent,
    spent_after,
    created_after,
    spent_before,
    created_before,
):
    """
    Compare results from Kupo API and direct DBSync query.
    Ensures that dbsync implementation matches Kupo behavior.
    """
    dbsync_url = os.environ.get("DBSYNC_HOST")
    if not dbsync_url:
        pytest.skip("DBSYNC_HOST not set")

    if spent_after is None and created_after is None:
        pytest.skip("Not having spent_after or created_after takes too long.")

    # Validation logic mirroring test_get_matches.py
    if spent and unspent:
        pytest.xfail("Cannot have both spent and unspent")
    if spent and (spent_after is not None or spent_before is not None):
        pytest.xfail("Cannot have spent and spent_after or spent_before")
    if unspent and (spent_after is not None or spent_before is not None):
        pytest.xfail("Cannot have unspent and spent_after or spent_before")
    if spent_after is not None and created_after is not None:
        pytest.xfail("Cannot have spent_after and created_after")
    if spent_before is not None and created_before is not None:
        pytest.xfail("Cannot have spent_before and created_before")

    try:
        # Get from Kupo
        kupo_res = kupo_client.get_matches(
            pattern=pattern,
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )

        # Get from DBSync
        dbsync_res = get_matches_dbsync(
            pattern=pattern,
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )

        # Sort matches for comparison
        k_matches = sorted(
            kupo_res.root, key=lambda m: (m.transaction_id, m.output_index)
        )
        d_matches = sorted(
            dbsync_res.root, key=lambda m: (m.transaction_id, m.output_index)
        )

        assert len(k_matches) != 0, "No matches found"

        assert (
            abs(len(k_matches) - len(d_matches)) <= 10
        ), f"Count mismatch: Kupo={len(k_matches)}, DBSync={len(d_matches)}"

        k_dict = {(k.transaction_id, k.output_index): k for k in k_matches}
        d_dict = {(d.transaction_id, d.output_index): d for d in d_matches}

        # Get intersection of keys from k_dict and d_dict
        intersection_keys = set(k_dict.keys()) & set(d_dict.keys())

        for key in intersection_keys:
            k = k_dict[key]
            d = d_dict[key]

            # Using model_dump to compare
            k_set = set(k.model_dump(mode="json"))
            d_set = set(d.model_dump(mode="json"))
            assert k_set == d_set, f"Mismatch.\nKupo: {k}\nDBSync: {d}"

    except ValueError as e:
        # Unexpected ValueError for valid parameters
        raise e
