"""Feed parsers against sample messages copied from the exchanges' public API docs."""
import asyncio
import json

from scalper.feeds import phemex


class FakeWS:
    def __init__(self, messages):
        self.messages, self.sent = messages, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    def __aiter__(self):
        async def gen():
            for m in self.messages:
                yield json.dumps(m)
        return gen()


def collect_events(monkeypatch, module, stream, messages, n):
    ws = FakeWS(messages)
    monkeypatch.setattr(module.websockets, "connect", lambda *a, **k: ws)

    async def run():
        out = []
        async for ev in stream(["BTC"]):
            out.append(ev)
            if len(out) == n:
                return out
    return asyncio.run(run()), ws


def test_phemex_trades_and_book(monkeypatch):
    msgs = [
        {"error": None, "id": 1, "result": {"status": "success"}},
        {"sequence": 1, "symbol": "BTCUSDT", "type": "snapshot",
         "trades_p": [[1666856062351916300, "Sell", "20703.6", "0.669"]]},
        {"sequence": 2, "symbol": "BTCUSDT", "type": "incremental",
         "trades_p": [[1666856076819029800, "Sell", "20700.3", "0.649"], [1666856076819029900, "Buy", "20700.5", "0.1"]]},
        {"depth": 30, "symbol": "BTCUSDT", "type": "snapshot", "timestamp": 1666854171201355264, "sequence": 3,
         "orderbook_p": {"asks": [["20702.9", "0.718"], ["20703.9", "0.524"]], "bids": [["20701.3", "1.0"], ["20700.5", "1.622"]]}},
        {"depth": 30, "symbol": "BTCUSDT", "type": "incremental", "timestamp": 1666854173705089711, "sequence": 4,
         "orderbook_p": {"asks": [["20702.9", "0"]], "bids": [["20701.3", "0"], ["20701.8", "0.3"]]}},
    ]
    events, ws = collect_events(monkeypatch, phemex, phemex.phemex_stream, msgs, 4)
    assert {m["method"] for m in ws.sent} == {"trade_p.subscribe", "orderbook_p.subscribe"}
    assert events[0] == ("trade", "BTC", [1666856076819, 20700.3, 0.649, -1])  # snapshot replay skipped
    assert events[1][2][3] == 1
    assert events[2] == ("book", "BTC", [1666854171201, 20701.3, 1.0, 20702.9, 0.718])
    assert events[3] == ("book", "BTC", [1666854173705, 20701.8, 0.3, 20703.9, 0.524])
