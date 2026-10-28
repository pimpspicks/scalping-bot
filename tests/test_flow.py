from scalper.flow import Slot, depth


def test_slot_splits_taker_buys_and_sells():
    s = Slot()
    s.add_trade(100.0, 2.0, 1)
    s.add_trade(101.0, 1.0, -1)
    assert (s.buy_qty, s.sell_qty, s.buy_usdt, s.sell_usdt, s.trades, s.max_trade, s.last) == \
        (2.0, 1.0, 200.0, 101.0, 2, 200.0, 101.0)


def test_depth_bands_around_mid():
    bids = {99.95: 1.0, 99.6: 2.0, 99.0: 5.0}
    asks = {100.05: 1.0, 100.4: 3.0, 101.0: 5.0}
    bb, ba, b10, a10, b50, a50 = depth(bids, asks)
    assert (bb, ba) == (99.95, 100.05)
    assert b10 == 99.95 and a10 == 100.05  # within 0.1% of mid 100
    assert b50 == 99.95 + 99.6 * 2 and a50 == 100.05 + 100.4 * 3
