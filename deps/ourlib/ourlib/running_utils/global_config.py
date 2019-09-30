global_config = None


def set_global_config(global_config_):
    global global_config
    global_config = global_config_


def get_global_config():
    assert global_config is not None

    return global_config
