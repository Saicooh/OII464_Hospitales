import logging
import sys

def setup_logger():
    """
    Configures and returns a logger for console output with a clean format.
    """
    # Create a custom logger
    logger = logging.getLogger('HospitalExperimentLogger')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to send logs to the console (stdout)
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a simple format that only shows the message
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# Create a single logger instance to be imported by other modules
logger = setup_logger()