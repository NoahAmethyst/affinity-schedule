import logging
import sys
import coloredlogs

logger = logging.getLogger(name='affinity-schedule')
logger.propagate = False


def init_logger():
    if any(getattr(handler, "_affinity_schedule_handler", False) for handler in logger.handlers):
        return logger

    ## Setup logger color
    colored_formatter = coloredlogs.ColoredFormatter(
        fmt='[%(name)s] %(asctime)s %(funcName)s %(lineno)-3d  %(message)s',
        level_styles=dict(
            debug=dict(color='white'),
            info=dict(color='green'),
            warning=dict(color='yellow', bright=True),
            error=dict(color='red', bold=True, bright=True),
            critical=dict(color='black', bold=True, background='red'),
        ),
        field_styles=dict(
            name=dict(color='white'),
            asctime=dict(color='white'),
            funcName=dict(color='white'),
            lineno=dict(color='white'),
        )
    )

    ## Setup logger streamHandler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt=colored_formatter)
    console_handler._affinity_schedule_handler = True
    logger.addHandler(hdlr=console_handler)
    logger.setLevel(level=logging.DEBUG)
    return logger

# log to file
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
