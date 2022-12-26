import argparse
from utils.utils import *
from graph.graph_process.lastfm_star_data_process import LastFmStarDataset
from graph.graph_process.lastfm_graph import LastFmGraph
from graph.graph_process.yelp_data_process import YelpDataset
from graph.graph_process.yelp_graph import YelpGraph
from graph.graph_process.book_data_process import BookDataset
from graph.graph_process.book_graph import BookGraph
from graph.graph_process.movie_graph import MovieGraph
from graph.graph_process.movie_data_process import MovieDataset
from graph.graph_process.folkscope_graph import FolkscopeGraph
from graph.graph_process.folkscope_data_process import FolkscopeDataset
DatasetDict = {
        LAST_FM_STAR: LastFmStarDataset,
        YELP_STAR: YelpDataset,
        BOOK: BookDataset,
        MOVIE: MovieDataset,
        FOLKSCOPE: FolkscopeDataset
    }
GraphDict = {
    LAST_FM_STAR: LastFmGraph,
    YELP_STAR: YelpGraph,
    BOOK: BookGraph,
    MOVIE: MovieGraph,
    FOLKSCOPE: FolkscopeGraph
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK, MOVIE, FOLKSCOPE],
                        help='One of {LAST_FM_STAR, YELP_STAR, BOOK, MOVIE, FOLKSCOPE}.')
    args = parser.parse_args()
    if args.data_name == 'BOOK':
        kg = BookGraph()
        with open('./datasets/processed_data/book/kg.pkl', 'wb') as f:
            pickle.dump(kg, f)
        dataset = DatasetDict[args.data_name]()
        save_dataset(args.data_name, dataset)
        return
    elif args.data_name == 'MOVIE':
        kg = MovieGraph()
        with open('./datasets/processed_data/movie/kg.pkl', 'wb') as f:
            pickle.dump(kg, f)
        dataset = DatasetDict[args.data_name]()
        save_dataset(args.data_name, dataset)
        return

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

