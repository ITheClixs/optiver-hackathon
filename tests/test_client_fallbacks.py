"""The real client has never been run here. A call the bot can do without must not cost the session if it is missing."""

import logging
from types import SimpleNamespace

import pytest

from otk.adapter import LiveExchange


class Full:
    def __init__(self):
        self.deleted = []
        self.cleared = []
        self.amended = []

    def connect(self):
        pass

    def get_outstanding_orders(self, _instrument_id):
        return {1: SimpleNamespace(order_id=1), 2: SimpleNamespace(order_id=2)}

    def poll_new_trades(self, _instrument_id):
        return ["fill"]

    def poll_new_trade_ticks(self, _instrument_id):
        return ["tick"]

    def get_pnl(self, _valuations=None):
        return 12.5

    def amend_order(self, instrument_id, **kwargs):
        self.amended.append((instrument_id, kwargs))
        return "amended"

    def delete_order(self, instrument_id, *, order_id):
        self.deleted.append((instrument_id, order_id))
        return "deleted"

    def delete_orders(self, instrument_id):
        self.cleared.append(instrument_id)
        return "cleared"


def without(*names):
    client = Full()

    class Limited:
        def __getattr__(self, name):
            if name in names:
                raise AttributeError(name)
            return getattr(client, name)

    exchange = LiveExchange(Limited)
    exchange.connect()
    return exchange, client


def test_the_full_client_is_used_as_it_is():
    exchange, client = without()
    assert exchange.poll_new_trades("A") == ["fill"]
    assert exchange.poll_new_trade_ticks("A") == ["tick"]
    assert exchange.get_pnl() == 12.5
    assert exchange.amend_order("A", order_id=1, volume=2) == "amended"
    assert exchange.delete_order("A", order_id=1) == "deleted"
    assert exchange.delete_orders("A") == "cleared"
    assert client.amended and client.deleted and client.cleared


@pytest.mark.parametrize("name", ["poll_new_trades", "poll_new_trade_ticks"])
def test_no_polling_call_means_nothing_new(name):
    exchange, _client = without(name)
    assert getattr(exchange, name)("A") == []


def test_no_pnl_call_reads_as_zero_and_says_so_once(caplog):
    exchange, _client = without("get_pnl")
    with caplog.at_level(logging.ERROR):
        assert exchange.get_pnl() == 0.0
        assert exchange.get_pnl() == 0.0
    assert sum("get_pnl" in record.message for record in caplog.records) == 1


def test_no_amend_call_deletes_and_the_quote_manager_places_the_right_size_next_pass():
    exchange, client = without("amend_order")
    exchange.amend_order("A", order_id=7, volume=2)
    assert client.deleted == [("A", 7)]


def test_no_single_delete_call_clears_the_instrument():
    exchange, client = without("delete_order")
    exchange.delete_order("A", order_id=7)
    assert client.cleared == ["A"]


def test_no_clear_call_deletes_each_order_so_the_kill_still_works():
    exchange, client = without("delete_orders")
    exchange.delete_orders("A")
    assert client.deleted == [("A", 1), ("A", 2)]
