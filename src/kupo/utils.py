# noqa

from kupo.models import Point

ERA_BOUNDARIES = {
    "last_byron_block": Point(
        slot_no=4492799,
        header_hash="f8084c61b6a238acec985b59310b6ecec49c0ab8352249afd7268da5cff2a457",
    ),
    "last_shelley_block": Point(
        slot_no=16588737,
        header_hash="4e9bbbb67e3ae262133d94c3da5bffce7b1127fc436e7433b87668dba34c354a",
    ),
    "last_allegra_block": Point(
        slot_no=23068793,
        header_hash="69c44ac1dda2ec74646e4223bc804d9126f719b1c245dadc2ad65e8de1b276d7",
    ),
    "last_mary_block": Point(
        slot_no=39916796,
        header_hash="e72579ff89dc9ed325b723a33624b596c08141c7bd573ecfff56a1f7229e4d09",
    ),
    "last_alonzo_block": Point(
        slot_no=72316796,
        header_hash="c58a24ba8203e7629422a24d9dc68ce2ed495420bf40d9dab124373655161a20",
    ),
}
