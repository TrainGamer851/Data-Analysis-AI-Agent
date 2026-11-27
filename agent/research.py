"""Opt-in, fixed-source public indicator research. Never sends dataset contents."""
import json
import math
import re
from datetime import date
from urllib.request import Request, build_opener, HTTPRedirectHandler
from agent.debug_logger import debug_log, DebugTimer
from agent.time_budget import request_timeout

INDICATORS = {
    'inflation': ('FP.CPI.TOTL.ZG', 'Consumer price inflation', '% annually'),
    'gdp growth': ('NY.GDP.MKTP.KD.ZG', 'Real GDP growth', '% annually'),
    'unemployment': ('SL.UEM.TOTL.ZS', 'Unemployment (modeled ILO estimate)', '% of labor force'),
}
COUNTRIES = {'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada', 'DE': 'Germany', 'FR': 'France', 'IN': 'India', 'JP': 'Japan', 'AU': 'Australia', 'MX': 'Mexico', 'BR': 'Brazil'}
MAX_BYTES = 256_000

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_indicator(country, indicator):
    if country not in COUNTRIES or indicator not in INDICATORS:
        raise ValueError('Unsupported research selection')
    code, _, _ = INDICATORS[indicator]
    url = f'https://api.worldbank.org/v2/country/{country}/indicator/{code}?format=json&per_page=20'
    timeout = request_timeout(8, reserve=5, minimum=1)
    debug_log('RESEARCH', f'Requesting World Bank indicator {code} for {country}')
    with build_opener(NoRedirect()).open(Request(url, headers={'Accept': 'application/json'}), timeout=timeout) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Research response too large')
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
        raise ValueError('Unexpected research response')
    observations = []
    for row in payload[1]:
        if not isinstance(row, dict):
            continue
        if row.get('indicator', {}).get('id') != code or row.get('country', {}).get('id') != country:
            continue
        year, value = row.get('date'), row.get('value')
        if not isinstance(year, str) or not re.fullmatch(r'\d{4}', year) or not 1900 <= int(year) <= date.today().year:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            continue
        observations.append((int(year), float(value)))
    if not observations:
        raise ValueError('No verified observation returned')
    year, value = max(observations)
    return year, value


def is_research_request(question):
    return isinstance(question, str) and bool(re.match(r"^(?:please\s+)?research\s+", question.strip(), re.I))


def try_research(question):
    q = question.strip().rstrip('?.!').strip()
    q = re.sub(r'^please\s+', '', q, flags=re.I)
    if not is_research_request(q):
        return None
    match = re.fullmatch(r'research (inflation|gdp growth|unemployment) for ([a-z]{2})', q, re.I)
    if not match or match[2].upper() not in COUNTRIES:
        return {'answer': 'Supported research: Research inflation for US; Research GDP growth for US; Research unemployment for US. Country codes: ' + ', '.join(COUNTRIES) + '.', 'table': None, 'error': None}
    indicator, country = match[1].lower(), match[2].upper()
    code, label, unit = INDICATORS[indicator]
    timer = DebugTimer('RESEARCH')
    try:
        year, value = fetch_indicator(country, indicator)
        answer = f'**External research — World Bank**\n\n{label}, {COUNTRIES[country]} ({year}): **{value:,.2f}{unit}**.\n\nSource: [World Bank indicator {code}](https://data.worldbank.org/indicator/{code}?locations={country}). Retrieved {date.today().isoformat()}.\n\nThis is a national indicator, not a finding from your uploaded dataset. The latest available observation may lag the current year.'
        debug_log('RESEARCH', 'Validated indicator identity, country, year and numeric value')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        debug_log('RESEARCH', f'Research unavailable: {type(exc).__name__}')
        answer = 'External research is unavailable or returned no valid observation. No external value has been estimated. Your dataset analysis remains available.'
    finally:
        timer.finish('Research finished')
    return {'answer': answer, 'table': None, 'error': None}
