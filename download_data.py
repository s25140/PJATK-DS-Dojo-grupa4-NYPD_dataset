# import requests
# import pandas as pd
# from tqdm import tqdm
#
# import requests
#
# socrata_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
#
# data_len = 8914838
#
# limit = data_len//100
# offset = 0
# all_data = []
#
# data_len = 8914838
#
#
#
# params = {
#     "$limit": data_len,
#     "$offset": offset,
# }
# response = requests.get(socrata_url, params=params)
#
#
# data = response.json()
#
# df = pd.DataFrame(data)

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

socrata_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
data_len = 8914838
batch_size = 100000


def fetch_data(offset, limit):
    params = {
        "$limit": limit,
        "$offset": offset,
    }
    response = requests.get(socrata_url, params=params)
    response.raise_for_status()
    return response.json()


def main():
    offsets = range(0, data_len, batch_size)
    all_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_data, offset, batch_size) for offset in offsets]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data = future.result()
                all_data.extend(data)
            except Exception as e:
                print(f"An error occurred: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv('nyc_data.csv', index=False)
    print("Data has been saved to nyc_data.csv")


if __name__ == "__main__":
    main()


