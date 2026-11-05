import pytest
from trade_actions import process_natural_chat_trade_command, execute_close_trade

def test_natural_chat_close_command_mock():
    class MockLogger:
        def __init__(self):
            self.closed = []
        def get_open_trades(self):
            return [{"id": 99, "ticker": "DIVISLAB.NS", "order_id": "TEST-1", "entry_price": 9664.0, "quantity": 2, "current_price": 10126.0, "status": "OPEN", "stage": "STAGE_PROFIT_LOCKED"}]
        def close_trade(self, order_id, exit_price):
            self.closed.append((order_id, exit_price))
    
    ml = MockLogger()
    res = process_natural_chat_trade_command("sell my position", trade_logger=ml)
    assert res is not None
    assert "Position Closed & Profit Banked" in res
    assert len(ml.closed) == 1

def test_regex_matching_patterns():
    from trade_actions import process_natural_chat_trade_command
    class MockLogger:
        def get_open_trades(self):
            return []
    
    ml = MockLogger()
    res = process_natural_chat_trade_command("sell DIVISLAB", trade_logger=ml)
    assert "no active open positions" in res.lower()
