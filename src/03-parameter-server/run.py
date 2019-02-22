import argparse
from time import sleep

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--job_and_task', type=str, required=True)
FLAGS, _ = parser.parse_known_args()
job_and_task = FLAGS.job_and_task

if ':' not in job_and_task:
    print('Use --job_and_task="job_name:task_index" notation.')
    exit(1)

job_name, task_id = job_and_task.split(':', maxsplit=2)
task_id = int(task_id)

# Cluster defines all jobs and where they are located, here we say that
# we have one 'parameter_server' job and two 'worker' jobs.
cluster = tf.train.ClusterSpec({
    'parameter_server': ['localhost:2222'],
    'worker': ['localhost:2223', 'localhost:2224'],
})
server = tf.train.Server(cluster, job_name=job_name, task_index=task_id)

with tf.device(f'/job:parameter_server/task:0'):
    # Parameter server holds our single value "a" with default value of 1.
    a = tf.Variable(1)

if job_name == 'parameter_server':

    # Parameter server will wait infinitely for updates made by workers.
    server.join()

elif job_name == 'worker':

    # Use local worker for all the worker operations.
    with tf.device(
        tf.train.replica_device_setter(
            cluster=cluster, worker_device=f'/job:worker/task:{task_id}',
        )
    ):
        # Keep track how many iterations have we done, incrementing global step
        global_step = tf.train.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)

        # The actual model training logic for worker.
        # Here we are just adding "a" to itself and recording it back to
        # parameter servers.
        add_to_itself = a.assign_add(a)

    # MonitoredTrainingSession allows hooks to define when to stop
    # the distributed workloads. Here we stop all workers when
    # global step number has been incremented by total of 10 times.
    is_chief = (task_id == 0)
    hooks = [tf.train.StopAtStepHook(last_step=10)]
    with tf.train.MonitoredTrainingSession(server.target, is_chief=is_chief, hooks=hooks) as sess:
        while not sess.should_stop():
            sleep(1)
            result = sess.run([add_to_itself, increment_global_step])
            print(f'step: {result[1]}, a: {result[0]}')
