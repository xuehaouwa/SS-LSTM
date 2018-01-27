import numpy as np
from utils import norm_distance, calculate_distance, calculate_angle


def get_angular_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: traj_data [ped_ID, frame_ID, x-coord, y-coord]
    """

    ang_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    width, height = dimensions[0], dimensions[1]
    width_bound, height_bound = neighborhood_size / (width * 1.0), neighborhood_size / (height * 1.0)

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data)):
        if data[i][1] == frame_ID:
            ped_list.append(data[i, :])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    # ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        return ang_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-2], ped_list[pedIndex][-1]
                width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
                height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-2], ped_list[otherIndex][-1]
                dis = calculate_distance([current_x, current_y], [other_x, other_y])
                norm_dis = norm_distance(dis)
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
                cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))

                ang_map[cell_x, cell_y] += norm_dis
        #                o_map[cell_x + cell_y*grid_size] = 1

        return ang_map


def get_distance_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """

    dis_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    width, height = dimensions[0], dimensions[1]
    width_bound, height_bound = neighborhood_size / (width * 1.0), neighborhood_size / (height * 1.0)

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        dis_map = np.reshape(dis_map, [int(neighborhood_size / grid_size) * int(neighborhood_size / grid_size), ])
        return dis_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
                height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                dis = calculate_distance([current_x, current_y], [other_x, other_y])
                norm_dis = norm_distance(dis)
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
                cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))

                dis_map[cell_x, cell_y] += norm_dis
        #                o_map[cell_x + cell_y*grid_size] = 1

        dis_map = np.reshape(dis_map, [int(neighborhood_size / grid_size) * int(neighborhood_size / grid_size), ])

        return dis_map


def get_grid_map(frame_ID, ped_ID, dimensions, neighborhood_size, grid_size, data):
    """
    This function computes rectangular occupancy map for each pedestrian at each frame.
    This occupancy map is used in group level LSTM.
    params:
        frame_ID: frame No.
        ped_ID: each ped in frame_ID
        dimensions : This will be a list [width, height], size of frame
        neighborhood_size : Scalar value representing the size of neighborhood considered (32)
        grid_size : Scalar value representing the size of the grid discretization (4)
        data: data of pixel_pos.csv file, [frame_ID, ped_ID, y-coord, x-coord]
    """

    grid_map = np.zeros((int(neighborhood_size / grid_size), int(neighborhood_size / grid_size)))
    width, height = dimensions[0], dimensions[1]
    width_bound, height_bound = neighborhood_size / (width * 1.0), neighborhood_size / (height * 1.0)

    ped_list = []

    # search for all peds in the same frame
    for i in range(len(data[0])):
        if data[0][i] == frame_ID:
            ped_list.append(data[:, i])
    # reshape ped_list to [num of ped, 4], [frame_ID, ped_ID, y-coord, x-coord]
    ped_list = np.reshape(ped_list, [-1, 4])

    if len(ped_list) == 0:
        print('no pedestrian in this frame!')
    elif len(ped_list) == 1:
        print('only one pedestrian in this frame!')
        grid_map = np.reshape(grid_map, [int(neighborhood_size / grid_size) * int(neighborhood_size / grid_size), ])
        return grid_map
    else:
        for pedIndex in range(len(ped_list)):
            if ped_list[pedIndex][1] == ped_ID:
                current_x, current_y = ped_list[pedIndex][-1], ped_list[pedIndex][-2]
                width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
                height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
                current_index = pedIndex
        for otherIndex in range(len(ped_list)):
            if otherIndex != current_index:
                other_x, other_y = ped_list[otherIndex][-1], ped_list[otherIndex][-2]
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
                cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))

                grid_map[cell_x, cell_y] += 1
        #                o_map[cell_x + cell_y*grid_size] = 1

        grid_map = np.reshape(grid_map, [int(neighborhood_size / grid_size) * int(neighborhood_size / grid_size), ])

        return grid_map


def get_total_map(angular_map, distance_map):

    return angular_map + distance_map

