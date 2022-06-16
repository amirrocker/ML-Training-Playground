from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler

'''

jobstores = {

}
executors = {
    'default' : {'type' : 'threadpool', 'max_workers' : 20 },
    'processpool' : ProcessPoolExecutor(max_workers=5)
}
job_defaults = {
    'coalesce' : False,
    'max_instances' : 3
}

sched_method2 = BlockingScheduler()

@sched_method2.scheduled_job('interval', seconds=130)
def timed_job():
    print('this should be run every 130 seconds .... ')

# sched_method2.configure(gconfig=gconfig)
'''


def my_job():
    print("Hello Background Job .... ")


executors = {
    'default': {'type': 'threadpool', 'max_workers': 20},
    'processpool': ProcessPoolExecutor(max_workers=5)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 3
}

scheduler = BlockingScheduler()

scheduler.scheduled_job(my_job, 'interval', seconds=0.5)

scheduler.start()

print("sdjldsfkjds")
