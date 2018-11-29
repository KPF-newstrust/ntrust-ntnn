from datetime import datetime, timezone
import sys


def iso_str_to_date(str):
    tok = str.split('"')
    if len(tok) != 3:
        return datetime.now()

    try:
        return datetime.strptime(tok[1], '%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(sys.exc_info())
        return datetime.now()


def to_timestamp(dt):
    return float(dt.strftime('%s'))


def from_timestamp(ts):
    return datetime.fromtimestamp(ts, timezone.utc)


