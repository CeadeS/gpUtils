from subprocess import run
import subprocess
import os
from numpy import array
from tensorflow.python.client import device_lib

def get_gpu_utilization(gpu_num = None, verbose=False):
    """
    returns GPU utilizaztion in percent

    Parameters:
    ----------
    :param gpu_num: integer optional
        the number of the gpu whose utilization u want to know
        pass None if all
    :param verbose: bool, optinal
        if true, tells you index, name, bus_id and utilization of all gpu
    :return: array of ints
        array with the numbers of utilization

    """
    if gpu_num != None:
        check_num(gpu_num)
    if verbose:
        cmd = "nvidia-smi --query-gpu=index,gpu_name,gpu_bus_id,utilization.gpu,memory.used --format=csv"
        res = str(subprocess.check_output(cmd, shell=True))
        [print(a) for a in res.split('\\n')[:-1]]
    cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,nounits"
    res = str(subprocess.check_output(cmd, shell=True))
    res= res.split('\\n')
    if gpu_num== None:
        return array([list(map(int,a.split(','))) for a in res[1:-1]])
    else:
        return array(list(map(int,res[gpu_num+1].split(','))))


def get_busy_gpus(gpu_num = None, bool=False, verbose=False):
    """
    returns GPU busy index of all busy GPUS in arrayby default see return

    Parameters:
    ----------
    :param gpu_num: number, optional
        number of the gpu you want to know if busy
        pass None if all
    :param bool: bool, optional
        switch to return the results in boolean array
        default == False: array of numbers of busy gpus
    :param verbose: bool, optional
        prints the utilization of all gpus if True
    :return: array of type int
        returns an array of the numbers of busy gpus
        if bool is true: returns a bool array of free gpus
    """
    if gpu_num != None:
        check_num(gpu_num)
        return (get_gpu_utilization(gpu_num, verbose)>(1,1)).min()
    res = (get_gpu_utilization(gpu_num, verbose)>(1,1)).min(axis=1)
    if bool:
        return res
    return res.nonzero()[-1]

def get_free_gpus(gpu_num = None, bool=False, verbose=False):
    """
    returns GPU busy index of all free GPUS in array by default see return

    Parameters:
    ----------
    :param gpu_num: number, optional
        number of the gpu you want to know if free
        pass None if all
    :param bool: bool, optional
        switch to return the results in boolean array
        default == False: array of numbers of busy gpus
    :param verbose: bool, optional
        printsthe utilization of all gpus if True
    :return: array of type int
        returns an array of the numbers of free gpus
        if bool is true: returns a bool array of free gpus
    """
    if gpu_num != None:
        check_num(gpu_num)
        return (get_gpu_utilization(gpu_num, verbose)<(1,1)).min()
    res = (get_gpu_utilization(gpu_num, verbose)<(1,1)).min(axis=1)
    if bool:
        return res
    return res.nonzero()[-1]

def cuda_set_n_free_gpus(num_gpu = None, verbose=False):
    """
    Sets the CUDA_VISIBLE_DEVICES setting to all free GPUS by default - to num_gpu free GPUS when num_gpus is set
    Parameters:
    ----------
    :param num_gpu: integer in range [0,3], optional
        number of gpu you wish to make CUDA_VISIBLE
    :param verbose:
        prints what DEVICES are visible to tensorflow after setting CUDA_VISIBLE_DEVICES
    :return:
        Nothing
    """
    free = get_free_gpus()
    if num_gpu != None:
        check_num(num_gpu)
        if len(free) < num_gpu:
            raise ValueError("Not enough free GPUs available - only %d available - use count_free_gpus() or pass 'None' as num_gpu for maximum number"%len(free))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str,free)))
    if verbose:
        print(get_tf_visible_gpus())

def count_free_gpus():
    """
    Count your free cpus
    :return: int
    return the number of free gpus
    """
    return len(get_free_gpus())

def check_num(gpu_num):
    """
    Checks if a number satisfies conditions type int and range [0,3]
    :param gpu_num:
    :return: nothing
    :raises: Value Error: if a condition is not satisfied
    """
    if not isinstance( gpu_num, int ):
        raise ValueError('GPU Number must be of Type: int')
    if not (gpu_num > -1 and gpu_num < 4):
        raise ValueError('GPU Number must be in [0,3]')

def get_tf_visible_gpus(verbose = False):
    """
    See what Devices are visible to Tensorflow
    :param verbose: bool,optinal
        prints what devides are visible
    :return: List of strings
        returns a List of Tensorflow visible GPUs
    """
    local_device_protos = device_lib.list_local_devices()
    if verbose:
        [print(x.name) for x in local_device_protos if x.device_type == 'GPU']
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    print("Result: \n [gpu util [%]  gpu mem util [mb]] \n", get_gpu_utilization(None,True))
    print(get_busy_gpus(bool=True))
    print(get_free_gpus(bool=True))
    print(get_free_gpus(0))
    print(cuda_set_n_free_gpus())
    print(get_tf_visible_gpus())
