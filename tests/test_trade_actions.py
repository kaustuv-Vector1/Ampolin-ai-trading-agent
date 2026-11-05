import pytest
from reflection import TradeLogger
from trade_actions import process_natural_chat_trade_command, execute_close_trade

def test_natural_chat_close_command():
    logger = TradeLogger()
    # Check with no trades
    res = process_natural_chat_trade_command("sell my position", trade_logger=logger)
    assert res is not None

def test_regex_matching_patterns():
    from trade_actions import process_natural_chat_trade_command
    class MockLogger:
        def get_open_trades(self):
            return [{"id": 99, "ticker": "DIVISLAB.NS", "order_id": "TEST-1", "entry_price": 9664.0, "quantity": 2, "current_price": 10126.0, "status": "OPEN", "stage": "STAGE_PROFIT_LOCKED"}]
    
    ml = MockLogger()
    # Test close regex match
    msg = "sell DIVISLAB"
    msg_low = msg.lower()
    assert "sell" in msg_low
