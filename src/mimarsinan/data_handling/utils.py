import pickle 

def get_multiprocess_friendly_data_provider(data_provider):
    mp_data_provider = pickle.loads(pickle.dumps(data_provider))
    mp_data_provider.set_num_workers(0)

    return mp_data_provider