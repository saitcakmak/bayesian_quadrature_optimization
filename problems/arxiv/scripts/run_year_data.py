from problems.arxiv.process_raw_data import ProcessRawData

import argparse


if __name__ == '__main__':
    # python -m problems.arxiv.scripts.run_year_data
    parser = argparse.ArgumentParser()
    parser.add_argument('month', help='e.g. 23')
    args = parser.parse_args()
    month = args.month

    files = ProcessRawData.generate_filenames_month(2016, int(args.month))
    ProcessRawData.get_click_data(
        files,"problems/arxiv/data/2016_%s_processed_data.json" % month)