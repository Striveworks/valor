from valor_core.detection import _calculate_101_pt_interp


def test__calculate_101_pt_interp():
    # make sure we get back 0 if we don't pass any precisions
    assert _calculate_101_pt_interp([], []) == 0

    # get back -1 if all recalls and precisions are -1
    assert _calculate_101_pt_interp([-1, -1], [-1, -1]) == -1
