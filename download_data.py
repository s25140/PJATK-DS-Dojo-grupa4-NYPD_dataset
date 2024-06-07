import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import logging


socrata_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
data_len = 8914838
batch_size = 100000
lock = threading.Lock()

logging.basicConfig(filename='errors.log', level=logging.ERROR)


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

            # Use lock to safely write to the file
            with lock:
                df.to_csv('nyc_data.csv', mode='a', header=False, index=False)

            # If successful, break out of the retry loop
            break
        except Exception as e:
            logging.error(f"An error occurred on attempt {attempt + 1} for offset {offset}: {e}")
            time.sleep(2 ** attempt)


def main():
    offsets = range(0, data_len, batch_size)

    # Create a CSV file and write the header (if needed)
    response = requests.get(socrata_url, params={'$limit': 1, '$offset': 0})
    data = response.json()  # Fetch just one record to get the header
    pd.DataFrame(data).to_csv('nyc_data.csv', mode='w', index=False)


    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_data_and_save, offset, batch_size, 3) for offset in offsets]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
