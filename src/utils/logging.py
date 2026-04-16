import logging
from easy_logging import EasyFormatter

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
	"""Create logger using the EasyFormatter and return it."""
	logger = logging.getLogger(name)
	logger.setLevel(level)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(EasyFormatter())
		logger.addHandler(handler)

	return logger
