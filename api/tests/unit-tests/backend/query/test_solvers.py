from valor_api.backend import models
from valor_api.backend.query.solvers import _solve_graph


# TODO
def test_recursive_search():
    print()

    select_from = models.Dataset
    joins = _solve_graph(
        select_from=select_from,
        label_source=models.Prediction,
        tables={models.GroundTruth},
    )

    from sqlalchemy import select

    stmt = select(select_from)
    for join in joins:
        stmt = join(stmt)

    print(stmt)
