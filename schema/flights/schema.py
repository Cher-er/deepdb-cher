from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_flights_schema(csv_path):
    schema = SchemaGraph()
    schema.add_table(Table(table_name='flights',
                           primary_key=['f_flightno'],
                           attributes=['year_date', 'unique_carrier', 'origin', 'origin_state_abr', 'dest',
                                       'dest_state_abr', 'dep_delay', 'taxi_out', 'taxi_in', 'arr_delay', 'air_time',
                                       'distance'],
                           csv_file_location=csv_path.format('flights_origin'),
                           table_size=5000000,
                           sample_rate=1.0,
                           fd_list=[('origin', 'origin_state_abr'), ('dest', 'dest_state_abr')])
                     )
    return schema
