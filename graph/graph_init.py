import argparse
from utils.utils import *
from graph.graph_process.lastfm_data_process import LastFmDataset
from graph.graph_process.lastfm_star_data_process import LastFmStarDataset
from graph.graph_process.lastfm_graph import LastFmGraph
from graph.graph_process.yelp_data_process import YelpDataset
from graph.graph_process.yelp_graph import YelpGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    args = parser.parse_args()
    DatasetDict = {
        LAST_FM: LastFmDataset,
        LAST_FM_STAR: LastFmStarDataset,
        YELP: YelpDataset,
        YELP_STAR: YelpDataset,
    }
    GraphDict = {
        LAST_FM: LastFmGraph,
        LAST_FM_STAR: LastFmGraph,
        YELP: YelpGraph,
        YELP_STAR: YelpGraph,
    }

    # Create 'data_name' instance for data_name.
    print('Load', args.data_name, 'from file...')
    print(RAW_DATA_DIR[args.data_name])
    if not os.path.isdir(RAW_DATA_DIR[args.data_name]):
        os.makedirs(RAW_DATA_DIR[args.data_name])
    dataset = DatasetDict[args.data_name](RAW_DATA_DIR[args.data_name])
    save_dataset(args.data_name, dataset)
    print('Save', args.data_name, 'dataset successfully!')

    # Generate graph instance for 'data_name'
    print('Create', args.data_name, 'graph from data_name...')
    dataset = load_dataset(args.data_name)
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    print('Save', args.data_name, 'graph successfully!')


if __name__ == '__main__':
    main()

