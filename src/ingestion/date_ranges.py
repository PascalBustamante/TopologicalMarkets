from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True, slots=True)
class DateRange:
    start: datetime
    end: datetime

    @classmethod
    def explicit(cls, start: datetime, end: datetime) -> "DateRange":
        return cls(start=start, end=end)

    @classmethod
    def single_day(cls, day: date) -> "DateRange":
        start = datetime(day.year, day.month, day.day)
        end = start + timedelta(days=1)
        return cls(start=start, end=end)

    @classmethod
    def last_n_days(cls, n: int, *, from_date: date | None = None) -> "DateRange":
        """Calendar days, not trading days — Alpaca just returns nothing for closed days."""
        anchor = from_date or date.today()
        end = datetime(anchor.year, anchor.month, anchor.day)
        start = end - timedelta(days=n)
        return cls(start=start, end=end)

    @classmethod
    def year_to_date(cls, *, year: int | None = None) -> "DateRange":
        yr = year or date.today().year
        start = datetime(yr, 1, 1)
        end = datetime.now() if yr == date.today().year else datetime(yr, 12, 31)
        return cls(start=start, end=end)

    @classmethod
    def month(cls, year: int, month: int) -> "DateRange":
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
        return cls(start=start, end=end)