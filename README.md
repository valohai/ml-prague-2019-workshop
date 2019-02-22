# SETUP

1. Install Docker if you don't have it already.

* Mac and Windows installers are included under `docker-installers`.
* If you want to download it on Mac: https://docs.docker.com/docker-for-mac/install/
* If you want to download it on Windows: https://hub.docker.com/editions/community/docker-ce-desktop-windows

2. Install Docker images

There are two .tar files in this directory;

* 'deepo-all-py36-cpu.tar' (3GB) contains all common machine learning frameworks like TensorFlow, Keras and PyTorch.
* 'horovod-tf18-py35.tar' (4GB) contains TensorFlow and Horovod

If you are tight on space on your laptop, you may just install the "deepo" image and not run the last Horovod example.

You don't need to copy these tar files to your laptop if you are low on space, you can install them from your USB stick, either way works.

```
docker load -i path/to/deepo-all-py36-cpu.tar
# or if you don't have the image tar...
# docker pull ufoym/deepo:all-py36-cpu

docker load -i horovod-tf18-py35.tar
# or if you don't have the image tar...
# docker pull uber/horovod:0.12.1-tf1.8.0-py3.5
```

3. Make a new "workshop" directory somewhere and copy `src/`, `ruksi-prague-workshop.pdf` and `README.md` there. Or everything if you wish, but will take longer ;)!

4. You are ready to go, you don't need the USB stick anymore so pass it forward.

# ASSIGNMENTS

`src/` has all source code and example datasets and we use it as s read-only directory inside the containers. There are assignments 01,02... inside the `src/` directory each discussing part of the days distributed learning agenda.

Most of the assignments in this workshop are to run specific scripts, see the output, read the code and understand what is happening. Reading the code is essential as there are comments explaining what is happening.

All of the following command are to be run in this directory.

```
cd path/to/this/directory
```

## 01-warmup

The first assignment focuses explaining low-level TensorFlow interface that is required to understand other examples.

Run the following command in this directory.

```
docker run --rm -it -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/01-warmup/basics.py
```

## 02-distributed

Introduction to low-level TensorFlow distributed learning basics, servers.

```
docker run --rm -it -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/02-distributed/basics.py
```

## 03-parameter-server

Here we start 1 parameter server with 2 workers.

Start each of these commands in this order in separate terminals to get the final output.

Remember to always navigate back to this directory when you open a new terminal.

```
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/03-parameter-server/run.py --job_and_task parameter_server:0
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/03-parameter-server/run.py --job_and_task worker:1
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/03-parameter-server/run.py --job_and_task worker:0
```

## 04-two-parameter-servers

Run 2 parameter servers with 2 workers, meaning one more parameter server than on the previous example.

Once again, start each of these commands in separate terminals.

```
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/04-two-parameter-servers/run.py --job_and_task parameter_server:1
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/04-two-parameter-servers/run.py --job_and_task parameter_server:0
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/04-two-parameter-servers/run.py --job_and_task worker:1
docker run --rm -it --network host -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/04-two-parameter-servers/run.py --job_and_task worker:0
```

## 05-mnist

Now we are getting to actually train a neural network! Run the following command and see how your neural network trains.

We are covering this so we can later compare how much code changes are needed to distribute this training on multiple processes, devices or machines.

This training should work out of the box.

```
docker run --rm -it -v $PWD/src:/src:ro ufoym/deepo:all-py36-cpu python /src/05-mnist/train.py
```

## 06-ring

In the final exercise we'll try out Horovod a little bit. As we won't have time to setup multiple servers for everybody, we'll do this locally in a single process and focus on the code changes needed to write distributed code in Horovod (which is easy).

So this is basically the same code as in 05, but now we want to utilize Horovod for optimization thus we assert that it is being used.

Quiz: But oh no, it won't even run! There are 4 commented lines of code that need to be uncommented to make this work on Horovod.

```
docker run --rm -it -v $PWD/src:/src:ro uber/horovod:0.12.1-tf1.8.0-py3.5 bash -c 'ldconfig /usr/local/cuda/lib64/stubs; mpirun -np 1 python /src/06-ring/train.py'
```

# TIPS

If one of your TensorFlow jobs freezes, just close the terminal window and run the following to kill all containers:

```
docker rm -f `docker ps -aq`
```

If you want to have freeform access to a container, you can just run `bash` in the container:

```
docker run --rm -it ufoym/deepo:all-py36-cpu bash
```

# Sources

- https://github.com/aymericdamien/TensorFlow-Examples/
- https://github.com/horovod/horovod
