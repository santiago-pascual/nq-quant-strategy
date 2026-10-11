from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_cme_snapshot import validate
from src.paper.cme_calendar import CMECalendarSnapshot, CMETradingCalendar, CalendarUnavailable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'src/paper/config/cme_mnq_calendar_2026-11-01_2026-11-03'


def test_official_review_identity_and_bounded_coverage():
    result = validate(BASE.with_suffix('.json'), Path(str(BASE)+'.review.json'))
    assert result['valid'] is True
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(BASE.with_suffix('.json')))
    with pytest.raises(CalendarUnavailable):
        calendar.snapshot.session_for_rth_date(date(2026, 11, 4))


def test_sunday_reopen_after_dst_and_regular_maintenance():
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(BASE.with_suffix('.json')))
    assert calendar.trading_date_for_timestamp(pd.Timestamp('2026-11-01T22:59Z')) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp('2026-11-01T23:00Z')) == date(2026,11,2)
    assert calendar.trading_date_for_timestamp(pd.Timestamp('2026-11-02T21:59Z')) == date(2026,11,2)
    assert calendar.trading_date_for_timestamp(pd.Timestamp('2026-11-02T22:00Z')) is None
    assert calendar.trading_date_for_timestamp(pd.Timestamp('2026-11-02T23:00Z')) == date(2026,11,3)


@pytest.mark.parametrize('interval', ['2026-11-01_2026-11-24', '2026-11-28_2026-11-30'])
def test_extended_reviews_preserve_thanksgiving_gap(interval):
    stem = ROOT/'src/paper/config'/f'cme_mnq_calendar_{interval}'
    assert validate(Path(str(stem)+'.json'), Path(str(stem)+'.review.json'))['valid']
    calendar = CMETradingCalendar(CMECalendarSnapshot.from_json(Path(str(stem)+'.json')))
    with pytest.raises(CalendarUnavailable):
        calendar.snapshot.session_for_rth_date(date(2026,11,26))
