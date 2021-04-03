from  matplotlib import pyplot as plt
import pptk
import pandas as pd
import json


def visualize_np_array_with_pptk(vis_data, look_at=(0., 0., 0.)):
    """
    Visualizes point-cloud data in numpy format
    Args:
        vis_data: numpy array to plot/view
        look_at: define camera position in coordinate system
    Returns: nothin
    """
    color_information = False  # default value
    point_data = []
    if vis_data.shape[1] > 3:  # data consists of x, y, z + additional information
        point_data = vis_data[:, :3]  # extracts coordinates
        color_information = True  # additional attributes available
    elif vis_data.shape[1] == 3:  # data only contains x, y, z coordinates
        point_data = vis_data
    else:  # data is not in desired format
        print("Data format of shape "+ str(vis_data.shape)+ " is not supported")
        exit(1)  # exit
    view = pptk.viewer(point_data)
    if color_information:  # add attributes if available
        color = [vis_data[:, i] for i in range(3, vis_data.shape[1])]
        view.attributes(*color)
    view.set(point_size=0.01)  # properties
    view.set(lookat=look_at)
    # view.set(phi=phi)
    # view.set(theta=theta)

    return 0


if __name__ == '__main__':
    dataset_G = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_gaussian/1558131117608556768.xyz'
    dataset_U = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_uniform/1558131117676654098.xyz'
    annot_data = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_annotation/1558131117608556768.json'

    #     Visualize Gausion Point Cloud
    data_G = pd.read_csv(dataset_G,names=[ 'x', 'y', 'z', 'intensity'], index_col=4)
    xyz_G = data_G.drop(['intensity'], axis=1)
    visualize_np_array_with_pptk(xyz_G)

    # Visualize Uniform Point Cloud
    data_U = pd.read_csv(dataset_U, names=['x', 'y', 'z', 'intensity','m','l','p'], index_col=4)
    data_U = data_U.drop(['m','l','p'], axis=1)
    xyz_U = data_U.drop(['intensity'], axis=1)
    visualize_np_array_with_pptk(xyz_U)

    #  Read Annotations
    f = open(annot_data)
    annotation = json.load(f)
    for i in annotation['labels']:
        print(i)
    print(annotation['label_type'])
    f.close()
    pass

    # visualize_np_array_with_pptk()
