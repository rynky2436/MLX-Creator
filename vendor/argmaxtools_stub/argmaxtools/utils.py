import logging
def get_logger(name=__name__):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg
