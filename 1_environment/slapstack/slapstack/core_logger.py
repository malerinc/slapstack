import os
from json import dump
from shutil import rmtree


from slapstack.core_state import State
from slapstack.helpers import create_folders
from slapstack.interface_templates import SlapLogger


class MatrixLogger(SlapLogger):
    def __init__(self, filepath):
        super().__init__(filepath)
        if filepath != '':
            self.log_dir = '/'.join(filepath.split('/')[:-1])
            if os.path.isdir(self.log_dir):
                rmtree(self.log_dir)
            create_folders(f'{self.log_dir}/dummy')
            # write logger object
            self.on = True
        else:
            self.on = False

    def log_state(self):
        if self.on:
            self.__write_heatmaps(self.slap_state)

    def __write_heatmaps(self, state: State):
        """
        Creates a json file with the information from the state matrices. The
        file can be used to create heatmap visualizations for the individual
        matrices. The json structure is as follows:
        {matrix_name_1: {
            data: nested list of values
            min_value: minimum value over the lists
            max_value: max value over the lists
            x_label: the column title
            y_label: the row title
            n_rows: the number of rows in the matrix
            n_cols: the number of columns in the matrix
            nfo_type: the information category; either 'jobs', 'tracking',
                'machines'
            }
            matrix_name_1: {
                ...
            }
            ...
        }
        :return:
        """
        f_name = f'{self.log_dir}/heatmaps/' \
                 f'{str(state.n_steps + state.n_silent_steps)}.json'
        f_matrices = open(create_folders(f_name), 'w')
        dump(state.to_dict(), f_matrices)
        f_matrices.close()
