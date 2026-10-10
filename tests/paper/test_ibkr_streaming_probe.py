from src.paper.ibkr_streaming_probe import streaming_confirmed


def test_delayed_stream_requires_mode_callback_and_valid_ticks():
    assert streaming_confirmed(market_data_type=3, valid_price_tick_count=1)
    assert not streaming_confirmed(market_data_type=3, valid_price_tick_count=0)
    assert not streaming_confirmed(market_data_type=1, valid_price_tick_count=4)
    assert not streaming_confirmed(market_data_type=None, valid_price_tick_count=4)
