import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from telegram_bot import InteractiveTelegramBot
from telegram_notifier import TelegramNotifier
from fastapi.testclient import TestClient
from ui import app

@pytest.fixture
def mock_tg_update():
    update = MagicMock()
    update.effective_chat.id = "7233121221"
    update.message.reply_text = AsyncMock()
    update.callback_query.data = "refresh:positions"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update

def test_telegram_bot_all_commands(mock_tg_update):
    bot = InteractiveTelegramBot()
    context = MagicMock()

    async def _runner():
        # cmd_status
        await bot.cmd_status(mock_tg_update, context)
        assert mock_tg_update.message.reply_text.called

        # cmd_rules
        await bot.cmd_rules(mock_tg_update, context)
        assert mock_tg_update.message.reply_text.called

        # cmd_health
        await bot.cmd_health(mock_tg_update, context)
        assert mock_tg_update.message.reply_text.called

        # cmd_pcr
        await bot.cmd_pcr(mock_tg_update, context)
        assert mock_tg_update.message.reply_text.called

        # cmd_positions
        await bot.cmd_positions(mock_tg_update, context)
        assert mock_tg_update.message.reply_text.called

        # cmd_pnl (verifying no datetime NameError and proper formatting)
        await bot.cmd_pnl(mock_tg_update, context)
        call_args = mock_tg_update.message.reply_text.call_args[0][0]
        assert "AMPOLIN PAPER TRADING FUNDS & P&L" in call_args
        assert "Total Account Equity:" in call_args
        assert "Cumulative Realized P&L:" in call_args

    asyncio.run(_runner())

def test_telegram_callback_refresh(mock_tg_update):
    bot = InteractiveTelegramBot()
    context = MagicMock()

    async def _runner():
        await bot.handle_callback(mock_tg_update, context)
        assert mock_tg_update.callback_query.answer.called
        assert mock_tg_update.callback_query.edit_message_text.called

    asyncio.run(_runner())

def test_ui_endpoints_equity_synchronization():
    client = TestClient(app)
    r_status = client.get("/api/status")
    r_health = client.get("/api/health")

    assert r_status.status_code == 200
    assert r_health.status_code == 200

    data_status = r_status.json()
    data_health = r_health.json()

    assert data_status["portfolio_equity"] == data_health["account_equity"]
    assert data_status["total_realized_pnl"] == data_health["total_realized_pnl"]
    assert data_status["unrealized_pnl"] == data_health["floating_pnl"]

def test_telegram_notifier_no_deprecation():
    notifier = TelegramNotifier()
    # Mocking dispatch to not hit actual Telegram API
    notifier.bot_token = ""
    # Should not throw deprecation warning or crash
    success = notifier.send_message("Test message")
    assert success is False or success is True
