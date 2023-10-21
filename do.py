import argparse
import logging
import time
import os
import shutil
import sys

from schema.flights.schema import gen_flights_schema
from data_preparation.prepare_single_tables import prepare_all_tables
from ensemble_creation.naive import create_naive_all_split_ensemble, naive_every_relationship_ensemble
from ensemble_creation.rdc_based import candidate_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)

    # generate hdf
    parser.add_argument('--generate_hdf', action='store_true')
    parser.add_argument('--csv_delimiter', default=',')
    parser.add_argument('--csv_path')
    parser.add_argument('--hdf_path')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=sys.maxsize)

    # generate ensembles
    parser.add_argument('--generate_ensemble', action='store_true')
    parser.add_argument('--ensemble_strategy', default='single')
    parser.add_argument('--ensemble_path')
    parser.add_argument('--pairwise_rdc_path', default=None)
    parser.add_argument('--samples_rdc_ensemble_tests', type=int, default=10000)
    parser.add_argument('--samples_per_spn', nargs='+', type=int, default=[10000000, 10000000, 2000000, 2000000])
    parser.add_argument('--post_sampling_factor', nargs='+', type=int, default=[30, 30, 2, 1])
    parser.add_argument('--rdc_threshold', type=float, default=0.3)
    parser.add_argument('--bloom_filters', action='store_true')
    parser.add_argument('--ensemble_budget_factor', type=int, default=5)
    parser.add_argument('--ensemble_max_no_joins', type=int, default=3)
    parser.add_argument('--incremental_learning_rate', type=int, default=0)
    parser.add_argument('--incremental_condition', type=str, default=None)

    # logging
    parser.add_argument('--log_level', type=int, default=logging.DEBUG,
                        help='DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50')

    args = parser.parse_args()

    # logging setting
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        filename="logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))
    )
    logger = logging.getLogger(__name__)

    # generate schema
    table_csv_path = args.csv_path + '/{}.csv'
    if args.dataset == 'flights':
        schema = gen_flights_schema(table_csv_path)
    else:
        raise ValueError('Dataset Unknown')

    # generate hdf files
    if args.generate_hdf:
        logger.info(f"Generating HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")

        if os.path.exists(args.hdf_path):
            logger.info(f"Removing target path {args.hdf_path}")
            shutil.rmtree(args.hdf_path)

        logger.info(f"Making target path {args.hdf_path}")
        os.makedirs(args.hdf_path)

        prepare_all_tables(schema, args.hdf_path, csv_seperator=args.csv_delimiter,
                           max_table_data=args.max_rows_per_hdf_file)
        logger.info(f"Files successfully created")

    # generate ensembles
    if args.generate_ensemble:
        if not os.path.exists(args.ensemble_path):
            os.makedirs(args.ensemble_path)

        if args.ensemble_strategy == 'single':
            create_naive_all_split_ensemble(schema, args.hdf_path, args.samples_per_spn[0], args.ensemble_path,
                                            args.dataset, args.bloom_filters, args.rdc_threshold,
                                            args.max_rows_per_hdf_file, args.post_sampling_factor[0],
                                            incremental_learning_rate=args.incremental_learning_rate)
        elif args.ensemble_strategy == 'relationship':
            naive_every_relationship_ensemble(schema, args.hdf_path, args.samples_per_spn[1], args.ensemble_path,
                                              args.dataset, args.bloom_filters, args.rdc_threshold,
                                              args.max_rows_per_hdf_file, args.post_sampling_factor[0],
                                              incremental_learning_rate=args.incremental_learning_rate)
        elif args.ensemble_strategy == 'rdc_based':
            logging.info(
                f"maqp(generate_ensemble: ensemble_strategy={args.ensemble_strategy}, incremental_learning_rate={args.incremental_learning_rate}, incremental_condition={args.incremental_condition}, ensemble_path={args.ensemble_path})")
            candidate_evaluation(schema, args.hdf_path, args.samples_rdc_ensemble_tests, args.samples_per_spn,
                                 args.max_rows_per_hdf_file, args.ensemble_path, args.database_name,
                                 args.post_sampling_factor, args.ensemble_budget_factor, args.ensemble_max_no_joins,
                                 args.rdc_threshold, args.pairwise_rdc_path,
                                 incremental_learning_rate=args.incremental_learning_rate,
                                 incremental_condition=args.incremental_condition)
        else:
            raise NotImplementedError


