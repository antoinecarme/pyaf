

import pyaf.Bench.TS_datasets as tsds
import model_esthetics_visualizer as viz

lDataset = tsds.load_ozone()

lVisualizer = viz.cModelEstheticsVisualizer()

lVisualizer.generate_video(lDataset)
