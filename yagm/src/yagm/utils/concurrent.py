import math
import multiprocessing as mp
import threading
from yagm.utils.shared_memory import SharedMemory

import numpy as np
import torch


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.stop_flag = threading.Event()

    def stop(self):
        self.stop_flag.set()

    @property
    def is_stopped(self):
        return self.stop_flag.is_set()

    # @override
    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.

        """
        try:
            if self._target:
                run_kwargs = self._kwargs.copy()
                run_kwargs["stop_flag"] = self.stop_flag
                self._target(*self._args, **run_kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs, run_kwargs


class StoppableProcess(mp.Process):
    """Process class with a safe_stop() method.
    The thread itself has to check regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableProcess, self).__init__(*args, **kwargs)
        self.stop_flag = mp.Event()

    def stop(self):
        self.stop_flag.set()

    @property
    def is_stopped(self):
        return self.stop_flag.is_set()

    # @override
    def run(self):
        """Method to be run in sub-process; can be overridden in sub-class"""
        if self._target:
            run_kwargs = self._kwargs.copy()
            run_kwargs["stop_flag"] = self.stop_flag
            self._target(*self._args, **run_kwargs)


# @TODO - implement __getstate__() and __setstate__() for pickle/unpickle\
# ref: https://stackoverflow.com/questions/74635994/pytorchs-share-memory-vs-built-in-pythons-shared-memory-why-in-pytorch-we
class ShmNdarray:
    def __init__(self, shm_name, shape, dtype, create=True):
        # should store primitive/lightweight attributes
        # so that it's easy/fast to be pickle/unpickle
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        if create:
            shm = SharedMemory(
                name=shm_name, create=True, size=self.nbytes, track=False
            )
            self.shm_name = shm.name

    @classmethod
    def from_numpy(cls, arr: np.ndarray, shm_name: str | None = None):
        shm = SharedMemory(name=shm_name, create=True, size=arr.nbytes, track=False)
        shm_name, shape, dtype = shm.name, arr.shape, arr.dtype
        # create a NumPy array backed by shared memory
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        # Copy the original data into shared memory
        # shm_arr[:] = arr[:]
        np.copyto(shm_arr, arr)
        # shm.close()
        return cls(shm_name, shape, dtype, create=False)

    def to_numpy(self):
        shm = SharedMemory(self.shm_name, track=False)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        # we need to keep at least 1 reference to this shared memory
        # as long as we access to returned numpy array
        # unless, SegmentationFault :)
        # @TODO (dangnh36): more convenient method to implement this feature
        #   - Add shm as attribute to numpy array in runtime: not easy as normal Python class since numpy implementation in C
        #   but it may be more complicated as i think with many side effects
        #   https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
        #   https://stackoverflow.com/questions/67509913/add-an-attribute-to-a-numpy-array-in-runtime
        return shm, arr

    def close(self):
        shm = SharedMemory(self.shm_name, track=False)
        shm.close()

    def unlink(self):
        shm = SharedMemory(self.shm_name, track=False)
        shm.unlink()


# @TODO - implement __getstate__() and __setstate__() for pickle/unpickle
# ref: https://stackoverflow.com/questions/74635994/pytorchs-share-memory-vs-built-in-pythons-shared-memory-why-in-pytorch-we
class ShmTensor:
    def __init__(self, shm_name, shape, dtype, create=True):
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = (
            math.prod(self.shape) * torch.tensor([], dtype=self.dtype).element_size()
        )

        if create:
            shared_mem = SharedMemory(
                name=shm_name, create=True, size=self.nbytes, track=False
            )
            self.shm_name = shared_mem.name  # Store actual shared memory name

    @classmethod
    def from_tensor(cls, tensor, shm_name=None):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        shm = SharedMemory(
            name=shm_name,
            create=True,
            size=tensor.numel() * tensor.element_size(),
            track=False,
        )
        shm_name, shape, dtype = shm.name, tensor.shape, tensor.dtype

        # Create a raw memory buffer and copy tensor data
        buffer = torch.frombuffer(shm.buf, dtype=dtype).reshape(shape)
        buffer.copy_(tensor)  # Copy data into shared memory

        return cls(shm_name, shape, dtype, create=False)

    def to_tensor(self):
        shm = SharedMemory(self.shm_name, track=False)
        tensor = torch.frombuffer(shm.buf, dtype=self.dtype).reshape(self.shape)
        return shm, tensor  # Must retain shared_mem reference

    def close(self):
        shm = SharedMemory(self.shm_name, track=False)
        shm.close()

    def unlink(self):
        shm = SharedMemory(self.shm_name, track=False)
        shm.close()
        shm.unlink()
