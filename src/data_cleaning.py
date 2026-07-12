import os
import re
import numpy as np
import pandas as pd

import logging
from pathlib import Path
import sys

project_root = Path().resolve().parent
log_file = project_root / "logs" / "app.log"
log_file.parent.mkdir(exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def fetch_event_duration(row):
    if pd.isna(row.event_start_date) or pd.isna(row.event_end_date):
        if '00.00.' == row.event_date_str[:6]:
            return row.event_date_str[6:]
        elif '00.' == row.event_date_str[:3]:
            return pd.to_datetime(row.event_date_str[3:], format='%m.%Y', errors='coerce').strftime('%B %Y')
        return 'Unknown'
    elif row.event_start_date == row.event_end_date:
        return 'One-day'
    else:
        return f'{(row.event_end_date-row.event_start_date).days + 1} days'

def split_event_dates(date_str):
    if pd.isna(date_str):
        return pd.Series([pd.NA, pd.NA])

    date_str = str(date_str)

    if len(date_str) == 10:
        start_date = end_date = date_str
        
    if '-' in date_str:
        start_date, end_date = date_str.split('-')

        if len(start_date) == 3:  # "01-03.05.2024"
            start_date = f"{start_date[:2]}.{'.'.join(end_date.split('.')[1:])}"

        elif len(start_date) == 6:  # "01.05-03.05.2024"
            start_date = f"{start_date[:5]}.{end_date.split('.')[-1]}"

    return pd.Series([start_date, end_date])


def extract_event_dates(data):
    # ensure it's a DataFrame column-safe input
    data = data.copy()

    # get unique values only once
    unique_dates = data['event_date_str'].dropna().unique()

    # build lookup table
    parsed = (
        pd.Series(unique_dates)
        .apply(split_event_dates)
    )

    parsed_data = parsed.rename(columns={0: 'event_start_date', 1: 'event_end_date'})#, 2: 'event_durations'})

    parsed_data['event_start_date'] = pd.to_datetime(parsed_data['event_start_date'], format='%d.%m.%Y', errors='coerce')
    parsed_data['event_end_date'] = pd.to_datetime(parsed_data['event_end_date'], format='%d.%m.%Y', errors='coerce')

    date_mask = parsed_data.event_start_date > parsed_data.event_end_date
    parsed_data.loc[date_mask, ['event_start_date', 'event_end_date']] = (
        parsed_data.loc[date_mask, ['event_end_date', 'event_start_date']].to_numpy()
    )
    parsed_data.insert(0, 'event_date_str', unique_dates)

    parsed_data['event_duration'] = parsed_data.apply(fetch_event_duration, axis=1)
    # map back using join (faster than merge)
    parsed_data = data[['event_date_str']].join(
            parsed_data.set_index('event_date_str'),
            on='event_date_str'
        )[['event_start_date', 'event_end_date', 'event_duration']]
    return parsed_data

def extract_event_name(event_name):

    unique_events = pd.DataFrame(event_name.unique(), columns=['event_info'])
    unique_events['event_name'] = unique_events.event_info.str.split('(').str[0].str.strip()
    unique_events['event_host_country'] = unique_events.event_info.str.split('(').str[-1].str.replace(')', '', regex=False).str.strip()
    
    event_detail = pd.merge(
        event_name.to_frame('event_info'),
        unique_events, how='left', on='event_info'
    )[['event_name', 'event_host_country']]
    return event_detail

def parse_timed_event_duration(d):
    """Convert timed event durations like '6h', '24h' to hours."""
    d = str(d)
    match = re.search(r'(\d+)(?::(\d+))?\s*h?', d)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2) or 0)
        return hours*60 + minutes 
    day_match = re.search(r'(\d+)\s*d', d)
    if day_match:
        return int(day_match.group(1)) * 24 * 60

    return np.nan

def parse_distance(d):
    """Extract numeric km distance from strings like '50km', '100mi', '6h'."""
    d = str(d).strip().lower()
    # Timed events (e.g. 6h, 24h) → flag as timed
    if re.search(r'\dh|d$', d) or 'hour' in d:
        return np.nan   # handled separately
    km_match = re.search(r'([\d.]+)\s*(km|k)\b', d, re.IGNORECASE)
    mi_match = re.search(r'([\d.]+)\s*(mi|m|mile|miles)\b', d, re.IGNORECASE)
    if km_match:
        return float(km_match.group(1))
    if mi_match:
        return round(float(mi_match.group(1)) * 1.60934, 2)
    try:
        return float(d)
    except:
        return np.nan

def parse_event_distance(event_dist):

    unique_distance = pd.DataFrame(event_dist.unique(), columns=['event_distance'])
    unique_distance.loc[
        unique_distance["event_distance"].str.contains("etappe", case=False, na=False),
        "event_type"
    ] = "Multi-Stage"
    event_type_mask = unique_distance.event_type.isna()
    unique_distance.loc[event_type_mask,'distance_km'] = unique_distance.event_distance.apply(parse_distance)
    event_mask = unique_distance.event_type.isna() & unique_distance.distance_km.isna()
    unique_distance.loc[
        event_mask,
        'timed_event_duration_min'
    ] = unique_distance.event_distance.apply(parse_timed_event_duration)
    unique_distance.loc[event_type_mask, "event_type"] = np.select(
        [
            unique_distance.loc[event_type_mask, "distance_km"].notna(),
            unique_distance.loc[event_type_mask, "timed_event_duration_min"].notna(),
        ],
        [
            "Distance",
            "Timed",
        ],
        default=""
    )

    distance_detail = pd.merge(
        pd.DataFrame(event_dist, columns=['event_distance']),
        unique_distance, how='left', on='event_distance'
    )[['event_type', 'distance_km', 'timed_event_duration_min']]
    return distance_detail

if __name__ == "__main__":
    print("This file is being run directly")