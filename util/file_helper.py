import codecs
import os
import errno
import shutil
from argparse import ArgumentParser


def mkdirs(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def dir_exists(dir_name):
    return os.path.exists(os.path.dirname(dir_name))


def delete_dir(dir_name):
    shutil.rmtree(dir_name)


def open_file_for_writing(file_name, encoding=None, mode='w', buffering=1):
    mkdirs(file_name)
    return codecs.open(file_name, mode, encoding, buffering=buffering)


def open_file_for_reading(file_name, encoding=None, mode='rb'):
    return codecs.open(file_name, mode, encoding)


def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-b", "--basepath", help="the base path where files will be written")
    parser.add_argument("-m", "--minEpsilon", help="minimum value of epsilon for exploration")
    parser.add_argument("-M", "--maxEpsilon", help="maximum value of epsilon for exploration")
    parser.add_argument("-N", "--nameExperiment", help="name of the experiment")

    args = parser.parse_args()
    base_path = vars(args)['basepath']
    min_eps = vars(args)['minEpsilon']
    max_eps = vars(args)['maxEpsilon']
    exp_name = vars(args)['nameExperiment']


    if not os.access(base_path, os.W_OK):
        raise Exception("cannot write to :'%s'; "
                        "specify the path where files will be written using the -b command line arg" % base_path)
    return base_path,min_eps,max_eps,exp_name



class WeightWriter:
    def __init__(self, base_path, filename):
        self._base_path = base_path
        self._weights_file = open_file_for_writing(self._base_path + filename, mode="a")

    def print(self, printer):
        printer(self._weights_file)
        self._flush(self._weights_file)

    def _flush(self, f):
        f.flush()
        os.fsync(f)

    def close(self):
        self._weights_file.close()

    @staticmethod
    def print_episode(base_path, episode_num, printer):
        if base_path is not None:
            base_path += 'weights/'
            weight_writer = WeightWriter(base_path=base_path, filename='weights-' + str(episode_num) + '.txt')
            weight_writer.print(printer)
            weight_writer.close()
