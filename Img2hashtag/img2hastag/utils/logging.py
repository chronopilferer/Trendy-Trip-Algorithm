import logging

def setup_logging(level=logging.INFO):
    if len(logging.getLogger().handlers) == 0:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
