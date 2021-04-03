import vispy
# from vispy.scene import visuals, SceneCanvas
import numpy as np
from vispy import app, visuals, scene
from  matplotlib import pyplot as plt
import pandas as pd




if __name__ == '__main__':
    dataset_G = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_gaussian/1558131117608556768.xyz'
    dataset_U = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_uniform/1558131117676654098.xyz'
    annot_data = '/media/ammar/HDD/LIDAR_datasets/volvo_dataset/Cirrus_dataset1/dataset1_annotation/1558131117608556768.json'


    data_G = pd.read_csv(dataset_G,names=[ 'x', 'y', 'z', 'intensity'], index_col=4)
    xyz_G = data_G.drop(['intensity'], axis=1)

    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 500



    colors = np.ones((38255, 4), dtype=np.float32)
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(xyz_G, face_color=colors, symbol='o', size=10,
                edge_width=0.5, edge_color='blue')
    app.run()
    pass
