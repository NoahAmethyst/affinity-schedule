import argparse
import logging
import agent
import signal

from util.logger import init_logger, logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please enter the configuration of the intelligent agent')

    parser.add_argument('-c', '--cores', type=int, default=1, help='please enter CPU usage(core)')
    parser.add_argument('-m', '--memory', type=int, default=1000, help='please enter Memory usage(GB)')
    parser.add_argument('-f', '--frequency', type=float, default=1.0, help='please enter communicate fequency(n/s)')
    parser.add_argument('-p', '--package', type=int, default=1, help='please enter package size(MB)')
    parser.add_argument('-t', '--target', type=str, default="", help='please enter communicate target(ip/service name)')
    parser.add_argument('-a', '--amount', type=int, default=1, help='please enter communicate amount(count in totally)')

    args = parser.parse_args()

    init_logger()

    logger.info(f'init args: {args}')
    logger.info("start to init the agent")

    my_agent = agent.Agent(args.cores, args.memory, args.frequency, args.package, args.target, args.amount, logger)

    signal.signal(signal.SIGTERM, my_agent.stop)

    my_agent.run()

    logger.info("end the agent")
