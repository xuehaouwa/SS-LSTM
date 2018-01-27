import os
import numpy as np
from occupancy_map import get_distance_map, get_grid_map


class DataProcesser:

    def __init__(self, data_dir, observed_frame_num, predicting_frame_num, dup_threshold):
        self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, 'pixel_pos.csv')
        self.dup_threshold = dup_threshold
        self.raw_data = None
        self.ped_num = None
        self.traj_data = []
        self.obs = []
        self.pred = []
        self.obs_length = observed_frame_num
        self.pred_length = predicting_frame_num

        self.from_csv()
        self.get_traj()
        self.get_obs_pred()

    def from_csv(self):
        print('Creating Raw Data from CSV file...')
        self.raw_data = np.genfromtxt(self.file_path, delimiter=',')
        self.ped_num = np.size(np.unique(self.raw_data[1, :]))

    def get_traj(self):
        """
        reshape data format from [frame_ID, ped_ID, y-coord, x-coord]
        to pedestrian_num * [ped_ID, frame_ID, x-coord, y-coord]
        """

        for pedIndex in range(self.ped_num):
            traj = []
            for i in range(len(self.raw_data[1])):
                if self.raw_data[1][i] == pedIndex + 1:
                    traj.append([self.raw_data[1][i], self.raw_data[0][i], self.raw_data[-1][i], self.raw_data[-2][i]])
            traj = np.reshape(traj, [-1, 4])

            if self.traj_filter(traj, dup_threshold=self.dup_threshold):
                self.traj_data.append(traj)

        return self.traj_data

    def get_obs_pred(self):
        """
        get input observed data and output predicted data
        """
        count = 0

        for pedIndex in range(len(self.traj_data)):

            if len(self.traj_data[pedIndex]) >= self.obs_length + self.pred_length:
                obs_pedIndex = []
                pred_pedIndex = []
                count += 1
                for i in range(self.obs_length):
                    obs_pedIndex.append(self.traj_data[pedIndex][i])
                for j in range(self.pred_length):
                    pred_pedIndex.append(self.traj_data[pedIndex][j + self.obs_length])

                obs_pedIndex = np.reshape(obs_pedIndex, [self.obs_length, 4])
                pred_pedIndex = np.reshape(pred_pedIndex, [self.pred_length, 4])

                self.obs.append(obs_pedIndex)
                self.pred.append(pred_pedIndex)

        self.obs = np.reshape(self.obs, [count, self.obs_length, 4])
        self.pred = np.reshape(self.pred, [count, self.pred_length, 4])

        return self.obs, self.pred

    def get_traj_input(self):
        traj_input = []
        for pedIndex in range(len(self.obs)):
            person_pedIndex = []
            for i in range(self.obs_length):
                person_pedIndex.append([self.obs[pedIndex][i][-2], self.obs[pedIndex][i][-1]])
            person_pedIndex = np.reshape(person_pedIndex, [self.obs_length, 2])

            traj_input.append(person_pedIndex)

        traj_input = np.reshape(traj_input, [len(self.obs), self.obs_length, 2])

        return traj_input

    def get_expected_output(self):
        expected_ouput = []
        for pedIndex in range(len(self.pred)):
            person_pedIndex = []
            for i in range(self.pred_length):
                person_pedIndex.append([self.pred[pedIndex][i][-2], self.pred[pedIndex][i][-1]])
            person_pedIndex = np.reshape(person_pedIndex, [self.pred_length, 2])

            expected_ouput.append(person_pedIndex)

        expected_ouput = np.reshape(expected_ouput, [len(self.pred), self.pred_length, 2])

        return expected_ouput

    def grid_map(self, neighborhood_size, dimensions, grid_size):
        grid_map_input = []

        for pedIndex in range(len(self.obs)):
            group_pedIndex = []
            for i in range(self.obs_length):
                o_map_pedIndex = get_distance_map(self.obs[pedIndex][i][1], self.obs[pedIndex][i][0], dimensions,
                                                               neighborhood_size, grid_size, self.raw_data)
                # o_map_pedIndex = np.reshape(o_map_pedIndex, [int(neighborhood_size / grid_size) ** 2, -1])
                group_pedIndex.append(o_map_pedIndex)
            group_pedIndex = np.reshape(group_pedIndex, [self.obs_length, int(neighborhood_size / grid_size) ** 2])

            grid_map_input.append(group_pedIndex)

        grid_map_input = np.reshape(grid_map_input,
                                       [-1, self.obs_length, int(neighborhood_size / grid_size) ** 2])

        return grid_map_input






