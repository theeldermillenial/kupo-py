"""Database sync implementation for Kupo matches."""

import binascii
import os
from typing import Any, List, Union

import dbsync  # type: ignore
from dotenv import load_dotenv
from pycardano import Address, AssetName, ScriptHash, TransactionId

from kupo.client import MatchResponse, Order
from kupo.models import (
    CreatedAt,
    Language,
    Match,
    Script,
    SpentAt,
    Value,
)

load_dotenv()

KUPO_PY_FROM_SLOT = os.getenv("KUPO_PY_FROM_SLOT", "0")


def _to_hex(value: Union[str, ScriptHash, AssetName, TransactionId]) -> str:
    if isinstance(value, (ScriptHash, AssetName, TransactionId)):
        return value.payload.hex()
    return value


def get_matches(
    pattern: str | Address,
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
    return_query: bool = False,
) -> Union[MatchResponse, tuple[str, list[Any]]]:
    """Get matches from dbsync directly."""

    kupo_start_slot = int(KUPO_PY_FROM_SLOT)

    if created_after is None:
        created_after = kupo_start_slot
    else:
        created_after = max(int(created_after), kupo_start_slot)

    # if spent_after is not None:
    #     spent_after = max(int(spent_after), kupo_start_slot)
    # elif spent:
    #     spent_after = kupo_start_slot

    with dbsync.session.get_session_context() as session:
        raw_conn = session.connection().connection
        cursor = raw_conn.cursor()

        try:
            # SELECT columns
            cols = [
                "address.address",
                "tx_out.value",
                "encode(tx_out.data_hash, 'hex')",
                "tx_out.index",
                "encode(tx.hash, 'hex') as tx_hash",
                "tx.block_index",
                "block.slot_no",
                "encode(block.hash, 'hex') as block_hash",
                # Spending info
                "encode(s_tx.hash, 'hex') as s_tx_hash",
                "s_block.slot_no as s_slot_no",
                "encode(s_block.hash, 'hex') as s_block_hash",
                "tx_out.consumed_by_tx_id as spent_input_id",
                # Assets
                """(
                    SELECT jsonb_object_agg(
                        concat(
                            encode(ma.policy, 'hex'),
                            '.',
                            encode(ma.name, 'hex')
                        ),
                        mato.quantity
                    )
                    FROM ma_tx_out mato
                    JOIN multi_asset ma ON mato.ident = ma.id
                    WHERE mato.tx_out_id = tx_out.id
                ) as assets""",
            ]

            if resolve_hashes:
                cols.extend(
                    [
                        "encode(datum.bytes, 'hex') as datum_bytes",
                        "encode(ref_script.bytes, 'hex') as script_bytes",
                        ("encode(ref_script.hash, 'hex') " "as script_hash_resolved"),
                        "ref_script.type as script_type",
                    ]
                )

            query = f"SELECT {', '.join(cols)} FROM tx_out"
            query += " JOIN tx ON tx_out.tx_id = tx.id"
            query += " JOIN block ON tx.block_id = block.id"
            query += " JOIN address ON tx_out.address_id = address.id"

            # Spending join
            query += " LEFT JOIN tx AS s_tx ON tx_out.consumed_by_tx_id = s_tx.id"
            query += " LEFT JOIN block AS s_block ON s_tx.block_id = s_block.id"

            # Pattern joins
            query += " LEFT JOIN stake_address "
            query += "ON tx_out.stake_address_id = stake_address.id"

            # Resolved joins
            if resolve_hashes:
                query += " LEFT JOIN datum ON tx_out.inline_datum_id = datum.id"
                query += " LEFT JOIN script AS ref_script "
                query += "ON tx_out.reference_script_id = ref_script.id"

            where_clauses = []
            params: List[Any] = []

            # Parse Pattern
            if isinstance(pattern, Address):
                where_clauses.append("address.address = %s")
                params.append(str(pattern))
            elif pattern == "*":
                pass  # No filter
            elif "/" in pattern:
                parts = pattern.split("/")
                cred = parts[0]
                stake = parts[1] if len(parts) > 1 else "*"

                if cred != "*":
                    try:
                        cred_bytes = binascii.unhexlify(cred)
                        where_clauses.append("address.payment_cred = %s")
                        params.append(cred_bytes)
                    except binascii.Error:
                        pass

                if stake != "*":
                    try:
                        stake_bytes = binascii.unhexlify(stake)
                        where_clauses.append("stake_address.hash_raw = %s")
                        params.append(stake_bytes)
                    except binascii.Error:
                        pass
            else:
                # Assume address string
                where_clauses.append("address.address = %s")
                params.append(pattern)

            # Filters
            if spent:
                where_clauses.append("tx_out.consumed_by_tx_id IS NOT NULL")
            elif unspent:
                where_clauses.append("tx_out.consumed_by_tx_id IS NULL")

            if created_after:
                where_clauses.append("block.slot_no > %s")
                params.append(int(created_after))
            if created_before:
                where_clauses.append("block.slot_no < %s")
                params.append(int(created_before))

            if spent_after:
                where_clauses.append("s_block.slot_no > %s")
                params.append(int(spent_after))
            if spent_before:
                where_clauses.append("s_block.slot_no < %s")
                params.append(int(spent_before))

            if transaction_id:
                where_clauses.append("encode(tx.hash, 'hex') = %s")
                params.append(_to_hex(transaction_id))

            if output_index is not None:
                where_clauses.append("tx_out.index = %s")
                params.append(output_index)

            # Asset filters
            if policy_id or asset_name:
                sub_clauses = ["m.tx_out_id = tx_out.id"]
                if policy_id:
                    sub_clauses.append("encode(a.policy, 'hex') = %s")
                    params.append(_to_hex(policy_id))
                if asset_name:
                    sub_clauses.append("encode(a.name, 'hex') = %s")
                    params.append(_to_hex(asset_name))

                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM ma_tx_out m "
                    f"JOIN multi_asset a ON m.ident = a.id "
                    f"WHERE {' AND '.join(sub_clauses)})"
                )

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            # Order
            if order == Order.DESC:
                query += (
                    " ORDER BY block.slot_no DESC, "
                    "tx.block_index DESC, tx_out.index DESC"
                )
            else:
                query += (
                    " ORDER BY block.slot_no ASC, "
                    "tx.block_index ASC, tx_out.index ASC"
                )

            if return_query:
                return query, params

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

            matches = []
            for row in rows:
                address = row[0]
                value_coins = row[1]
                datum_hash = row[2]
                out_index = row[3]
                tx_hash = row[4]
                tx_idx = row[5]
                slot = row[6]
                block_h = row[7]

                s_tx_hash = row[8]
                s_slot = row[9]
                s_block_h = row[10]
                spent_input_id = row[11]
                assets_json = row[12]

                # Optional resolved
                datum_bytes = None
                script_bytes = None
                script_hash_res = None
                script_type = None
                if resolve_hashes:
                    datum_bytes = row[13]
                    script_bytes = row[14]
                    script_hash_res = row[15]
                    script_type = row[16]

                # Construct Value
                assets = None
                if assets_json:
                    assets = assets_json

                val = Value(coins=value_coins, assets=assets)

                created_at = CreatedAt(slot_no=slot, header_hash=block_h)

                spent_at = None
                if spent_input_id is not None:
                    spent_at = SpentAt(
                        slot_no=s_slot,
                        header_hash=s_block_h,
                        transaction_id=s_tx_hash,
                        input_index=0,
                    )

                # Script Logic
                script = None
                if script_bytes and script_type:
                    lang = Language.native
                    if script_type == "plutusV1":
                        lang = Language.plutus_v1
                    elif script_type == "plutusV2":
                        lang = Language.plutus_v2
                    elif script_type == "plutusV3":
                        lang = Language.plutus_v3
                    script = Script(language=lang, script=script_bytes)

                matches.append(
                    Match(
                        transaction_index=tx_idx,
                        transaction_id=tx_hash,
                        output_index=out_index,
                        address=address,
                        value=val,
                        datum_hash=datum_hash,
                        datum=datum_bytes,  # hex string
                        script_hash=script_hash_res,
                        created_at=created_at,
                        spent_at=spent_at,
                        script=script,
                    )
                )

            return MatchResponse(root=matches)

        except Exception as e:
            raise e
        finally:
            cursor.close()
