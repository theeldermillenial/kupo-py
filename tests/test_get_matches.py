import pytest

from charli3_dendrite import MinswapCPPState


def test_get_all_matches(
    benchmark,
    kupo_client,
    spent,
    unspent,
    spent_after,
    created_after,
    spent_before,
    created_before,
):

    try:

        if spent_after is None and created_after is None:
            pytest.skip("Not having spent_after or created_after takes too long.")

        benchmark(
            kupo_client.get_all_matches,
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )
    except ValueError as e:
        if spent and unspent:
            pytest.xfail(f"Cannot have both spent and unspent in parameters: {e}")
        elif spent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have spent and spent_after or spent_before in parameters: {e}"
            )
        elif unspent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have unspent and spent_after or spent_before in parameters: {e}"
            )
        elif spent_after is not None and created_after is not None:
            pytest.xfail(
                f"Cannot have spent_after and created_after in parameters: {e}"
            )
        elif spent_before is not None and created_before is not None:
            pytest.xfail(
                f"Cannot have spent_before and created_before in parameters: {e}"
            )
        else:
            raise e


@pytest.mark.asyncio
async def test_get_all_matches_async(
    benchmark,
    kupo_client,
    spent,
    unspent,
    spent_after,
    created_after,
    spent_before,
    created_before,
):

    try:

        if spent_after is None and created_after is None:
            pytest.skip("Not having spent_after or created_after takes too long.")

        await kupo_client.get_all_matches_async(
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )
    except ValueError as e:
        if spent and unspent:
            pytest.xfail(f"Cannot have both spent and unspent in parameters: {e}")
        elif spent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have spent and spent_after or spent_before in parameters: {e}"
            )
        elif unspent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have unspent and spent_after or spent_before in parameters: {e}"
            )
        elif spent_after is not None and created_after is not None:
            pytest.xfail(
                f"Cannot have spent_after and created_after in parameters: {e}"
            )
        elif spent_before is not None and created_before is not None:
            pytest.xfail(
                f"Cannot have spent_before and created_before in parameters: {e}"
            )
        else:
            raise e


def test_get_matches(
    benchmark,
    kupo_client,
    pattern,
    spent,
    unspent,
    spent_after,
    created_after,
    spent_before,
    created_before,
):

    try:

        if spent_after is None and created_after is None:
            pytest.skip("Not having spent_after or created_after takes too long.")

        benchmark(
            kupo_client.get_matches,
            pattern=pattern,
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )
    except ValueError as e:
        if spent and unspent:
            pytest.xfail(f"Cannot have both spent and unspent in parameters: {e}")
        elif spent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have spent and spent_after or spent_before in parameters: {e}"
            )
        elif unspent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have unspent and spent_after or spent_before in parameters: {e}"
            )
        elif spent_after is not None and created_after is not None:
            pytest.xfail(
                f"Cannot have spent_after and created_after in parameters: {e}"
            )
        elif spent_before is not None and created_before is not None:
            pytest.xfail(
                f"Cannot have spent_before and created_before in parameters: {e}"
            )
        else:
            raise e


@pytest.mark.asyncio
async def test_get_matches_async(
    benchmark,
    kupo_client,
    pattern,
    spent,
    unspent,
    spent_after,
    created_after,
    spent_before,
    created_before,
):

    try:

        if spent_after is None and created_after is None:
            pytest.skip("Not having spent_after or created_after takes too long.")

        await kupo_client.get_matches_async(
            pattern=pattern,
            spent=spent,
            unspent=unspent,
            spent_after=spent_after,
            created_after=created_after,
            spent_before=spent_before,
            created_before=created_before,
        )
    except ValueError as e:
        if spent and unspent:
            pytest.xfail(f"Cannot have both spent and unspent in parameters: {e}")
        elif spent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have spent and spent_after or spent_before in parameters: {e}"
            )
        elif unspent and (spent_after is not None or spent_before is not None):
            pytest.xfail(
                f"Cannot have unspent and spent_after or spent_before in parameters: {e}"
            )
        elif spent_after is not None and created_after is not None:
            pytest.xfail(
                f"Cannot have spent_after and created_after in parameters: {e}"
            )
        elif spent_before is not None and created_before is not None:
            pytest.xfail(
                f"Cannot have spent_before and created_before in parameters: {e}"
            )
        else:
            raise e
