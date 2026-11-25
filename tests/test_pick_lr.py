from tools.pick_lr import choose_lr


def test_choose_lr():
    assert choose_lr(0.004) == 0.001
    assert choose_lr(0.012) == 0.0005
    assert choose_lr(-0.01) == 0.001
