class app:
    log_level = 'INFO'
    log_dir = 'logs'

    port = 7090

    model_dir = 'saved_model'

    class classifier:
        version = '3'

    class NER:
        version = '4'
        w2v_version = '1'

    class QER:
        version = '1'
