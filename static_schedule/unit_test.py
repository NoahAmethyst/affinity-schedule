import pytest

from affinity.affinity import cal_affinity
from static_schedule.best_fit_scheduler import BestFitScheduler
from static_schedule.first_fit_scheduler import FirstFitScheduler
from static_schedule.multi_stage_scheduler import MultiStageScheduler
from static_schedule.offline_scheduler import Scheduler
from static_schedule.worst_fit_scheduler import WorstFitScheduler
from util.logger import init_logger, logger

input_dir = '../data/input'
output_dir = '../data/plan'

init_logger()


def test_best_fit_schedule():
    logger.info('start calculating node and pod affinity')
    pod_affinity, node_affinity = cal_affinity(input_dir)
    logger.info('calculating node and pod affinity done,start generate static schedule plan with best fit proxy')
    scheduler = BestFitScheduler(input_dir, pod_affinity, node_affinity)
    ### schedule
    _plan = scheduler.schedule()

    ### check
    plan = scheduler.check_and_gen(scheduler, _plan)

    logger.info(f'generating plan has done')
    for single_plan in plan:
        logger.info(f'{single_plan.__dict__}')


def test_first_fit_schedule():
    logger.info('start calculating node and pod affinity')
    pod_affinity, node_affinity = cal_affinity(input_dir)
    logger.info('calculating node and pod affinity done,start generate static schedule plan with first fit proxy')
    scheduler = FirstFitScheduler(input_dir, pod_affinity, node_affinity)

    ### schedule
    _plan = scheduler.schedule()
    ### check
    plan = scheduler.check_and_gen(scheduler, _plan)

    logger.info(f'generating plan has done')
    for single_plan in plan:
        logger.info(f'{single_plan.__dict__}')


def test_multi_stage_fit_schedule():
    logger.info('start calculating node and pod affinity')
    pod_affinity, node_affinity = cal_affinity(input_dir)
    logger.info('calculating node and pod affinity done,start generate static schedule plan with multi stage fit proxy')
    scheduler = MultiStageScheduler(input_dir, pod_affinity, node_affinity)
    ### schedule
    _plan = scheduler.schedule(enable_draw=True)

    ### check
    ### check
    plan = scheduler.check_and_gen(scheduler, _plan)

    logger.info(f'generating plan has done')
    for single_plan in plan:
        logger.info(f'{single_plan.__dict__}')

    ### draw
    scheduler.draw('../data/others')


def test_worst_fit_schedule():
    logger.info('start calculating node and pod affinity')
    pod_affinity, node_affinity = cal_affinity(input_dir)
    logger.info('calculating node and pod affinity done,start generate static schedule plan with worst fit proxy')
    scheduler = WorstFitScheduler(input_dir, pod_affinity, node_affinity)
    ### schedule
    ### check
    plan = scheduler.check_and_gen(scheduler, pod_affinity)

    logger.info(f'generating plan has done')
    for single_plan in plan:
        logger.info(f'{single_plan.__dict__}')
