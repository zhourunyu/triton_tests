import logging
import sys

import acl

_ACL_DEBUG = 0
_ACL_INFO = 1
_ACL_WARNING = 2
_ACL_ERROR = 3

logger = logging.getLogger("acllite")

def log_error(*log_msg):
    """Recode error level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = "".join([str(i) for i in log_msg])
    logger.error(log_str)
    log_str = "[ERROR]\t" + log_str

    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_ERROR, message)

def log_warning(*log_msg):
    """Recode warning level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = "".join([str(i) for i in log_msg])
    logger.warning(log_str)
    log_str = "[WARNING]\t" + log_str

    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_WARNING, message)

def log_info(*log_msg):
    """Recode info level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = "".join([str(i) for i in log_msg])
    logger.info(log_str)
    log_str = "[INFO]\t" + log_str

    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_INFO, message)

def log_debug(*log_msg):
    """Recode debug level log to file
    Args:
        *log_msg: format string and args list
    """
    log_str = "".join([str(i) for i in log_msg])
    logger.debug(log_str)
    log_str = "[DEBUG]\t" + log_str

    caller_frame = sys._getframe().f_back
    # caller file
    filename = caller_frame.f_code.co_filename
    # caller line no
    line_no = caller_frame.f_lineno
    # caller function
    func_name = caller_frame.f_code.co_name

    message = "[" + filename + ":" + str(line_no) + \
              " " + func_name + "]" + log_str
    acl.app_log(_ACL_DEBUG, message)
