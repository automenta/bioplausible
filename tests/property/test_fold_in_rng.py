"""fold_in RNG properties (TODO11 R11.2.24).

The seed must be a pure function of its coordinates, injective over the
sampled coordinate grid, and inside ``torch.manual_seed``'s safe range.
"""

from hypothesis import given
from hypothesis import strategies as st

from computronium.core.system_trainer._resume import fold_in

coords = st.tuples(
    st.integers(min_value=0, max_value=2**31),
    st.integers(min_value=0, max_value=10_000),
    st.integers(min_value=0, max_value=100_000),
)


@given(coords)
def test_fold_in_is_pure(coordinate: tuple[int, int, int]) -> None:
    base, epoch, batch = coordinate
    assert fold_in(base, epoch, batch) == fold_in(base, epoch, batch)


@given(coords, coords)
def test_fold_in_is_coordinate_injective(
    a: tuple[int, int, int], b: tuple[int, int, int]
) -> None:
    assert (fold_in(*a) == fold_in(*b)) == (a == b)


@given(coords, st.integers(min_value=1, max_value=100))
def test_fold_in_domain_separates(
    coordinate: tuple[int, int, int], domain: int
) -> None:
    base, epoch, batch = coordinate
    assert fold_in(base, epoch, batch) != fold_in(base, epoch, batch, domain=domain)


@given(coords)
def test_fold_in_within_torch_seed_range(coordinate: tuple[int, int, int]) -> None:
    assert 0 <= fold_in(*coordinate) <= 2**63 - 1
