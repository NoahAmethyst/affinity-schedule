import pytest

from affinity.affinity import cal_affinity
from dynamic_schedule.main import dynamic_schedule
from static_schedule.unit_test import output_dir
from util.logger import logger


def test_dynamic_schedule():
    input_dir = '../data/input'
    logger.info('start calculating node and pod affinity')
    pod_affinity, _ = cal_affinity(input_dir)
    logger.info('calculating  pod affinity done,start generate dynamic schedule plan')
    plan = dynamic_schedule(input_dir, pod_affinity)
    logger.info('generating dynamic schedule plan done')
    print(f'dynamic scheduler plan is:{plan}')
