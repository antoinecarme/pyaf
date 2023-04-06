import glob
import imageio as iio
import numpy as np

import datetime
t = datetime.datetime.now()
lDateStr = t.strftime('%Y_%m_%d_%H_%M_%S')

def generate_video(video_type):
    images = []
    for f in glob.glob("outputs/*" + video_type + "*_output.png"):
        print("CREATE_TEST_MOVIE", f)
        lArray = iio.imread(f, pilmode="RGB").astype(np.uint8)
        images.append(lArray)
    max_shape = (max([x.shape[0] for x in images]), max([x.shape[1] for x in images]), 3)
    resized_images = []
    for im in images:
        img = im
        if(im.shape != max_shape):
            img = np.zeros(max_shape).astype(np.uint8)
            print(im.shape, img.shape)
            img[0:im.shape[0] , 0:im.shape[1]] = im[0:im.shape[0] , 0:im.shape[1]]
        resized_images.append(img)
        
    lName = "outputs/pyaf_all_tests_video_" + video_type + "_" + lDateStr + ".mp4"

    print("GENERATING_VIDEO_FILE", lName)
    
    writer = iio.get_writer(lName, fps=20)
    for im in resized_images:
        writer.append_data(im)
    writer.close()
    return video_type
    
import multiprocessing as mp

vtypes = "AR_decomp Cycle_decomp Forecast_decomp prediction_intervals quantiles TransformedForecast_decomp Trend_decomp".split()

pool = mp.Pool(processes=12)
for res in pool.imap_unordered(generate_video, vtypes):
    print("FINISHED_GENERATING_VIDEO_FOR_TYPE" , res)
pool.close()
pool.join()
del pool
