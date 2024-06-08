import numpy as np
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import logging

col_names = ['cmplnt_num', 'cmplnt_fr_dt', 'cmplnt_fr_tm', 'cmplnt_to_dt', 'cmplnt_to_tm', 'addr_pct_cd', 'rpt_dt',
             'ky_cd','ofns_desc', 'pd_cd', 'pd_desc', 'crm_atpt_cptd_cd', 'law_cat_cd', 'boro_nm', 'loc_of_occur_desc',
             'prem_typ_desc','juris_desc', 'jurisdiction_code', 'parks_nm', 'hadevelopt', 'housing_psa', 'x_coord_cd',
             'y_coord_cd','susp_age_group', 'susp_race', 'susp_sex', 'latitude', 'longitude', 'lat_lon', 'patrol_boro',
             'station_name','vic_age_group', 'vic_race','vic_sex', ':@computed_region_efsh_h5xi',
             ':@computed_region_f5dn_yrer', ':@computed_region_yeji_bk3q',':@computed_region_92fq_4b7q',
             ':@computed_region_sbqj_enih', 'transit_disrict']

socrata_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
data_len = 8914838
batch_size = 100000
lock = threading.Lock()

logging.basicConfig(filename='errors.log', level=logging.ERROR)


def preprocess_data(df):
    # There were `computed_region` columns, that I suppose came from API call, so I drop them at the start
    # Also, lat_lon - is redundant column, that I see from the beginning, will drop it too.
    df.drop([':@computed_region_efsh_h5xi',
             ':@computed_region_f5dn_yrer', ':@computed_region_yeji_bk3q', ':@computed_region_92fq_4b7q',
             ':@computed_region_sbqj_enih', 'lat_lon'], inplace=True, axis=1)

    # Change None valuaes to actual None, to reduce the memory(before DataSet was 3.7Gb, now 2.5Gb)
    df.replace('(null)', None, inplace=True)
    df.replace('UNKNOWN', None, inplace=True)
    return df


def fetch_data_and_save(offset, limit, max_retries=3):
    params = {
        "$limit": limit,
        "$offset": offset,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(socrata_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(data)
            df = preprocess_data(df)

            # Use lock to safely write to the file
            with lock:
                df.to_csv('data/nyc_data.csv', mode='a', header=False, index=False)

            # If successful, break out of the retry loop
            break
        except Exception as e:
            logging.error(f"An error occurred on attempt {attempt + 1} for offset {offset}: {e}")
            time.sleep(2 ** attempt)


def main():
    offsets = range(1, data_len, batch_size)

    response = requests.get(socrata_url, params={'$limit': 1, '$offset': 0})
    data = response.json()
    df = pd.DataFrame(data)
    df['transit_district'] = [None]
    preprocess_data(df).to_csv('data/nyc_data.csv', mode='w', index=False)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_data_and_save, offset, batch_size, 3) for offset in offsets]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
