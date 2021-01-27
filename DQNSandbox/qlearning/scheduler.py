from pytz import utc
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ProcessPoolExecutor

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

scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(my_job, 'interval', seconds=130)


