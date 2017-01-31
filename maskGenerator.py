#!/usr/bin/python
import os
import cv2
import sys
import motion_heatmap
import matplotlib.pyplot as plt


class backSub:

    def __init__(self, video_src, path):
        self.location = video_src
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.dir = path

    def run(self):
        frame_no = 0
        for file in os.listdir(self.location):
            if frame_no <= 20:
                # colored image denoising
                print('Background subsraction for input frame ' + str(frame_no))
                frame = cv2.imread(os.path.join(self.location, file))
                # cv2.imshow('frame',frame)
                frame = cv2.fastNlMeansDenoisingColored(
                    frame, None, 10, 10, 7, 21)
                fgmask = self.fgbg.apply(frame)
                cv2.imshow('frame', fgmask)
                cv2.waitKey(1)
                cv2.imwrite(os.path.join(
                    self.dir, str(frame_no) + '.png'), fgmask)
                frame_no += 1


class heatMap_Generator:

    def __init__(self, src_path, dst_path):
        self.images = sorted([
            src_path + '/' + f for f in os.listdir(src_path)
            if os.path.isfile(os.path.join(src_path, f)) and f.endswith('.png')
        ])
        self.finalname = 'motion_heatmap' + '.png'
        self.path = dst_path
        self.mh = motion_heatmap.MotionHeatmap(
            num_vertical_divisions=360,
            num_horizontal_divisions=480,
            images=self.images,
        )

    def run(self):
        self.mh.generate_motion_heatmap(self.path, self.finalname)


def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(source, backsubPath, heatMapPath):

    video_src = source
    path_to_backsub = backsubPath
    path_to_heatmap = heatMapPath

    makeDir(path_to_backsub)
    makeDir(path_to_heatmap)

    try:
        backSub(video_src, path_to_backsub).run()
        cv2.destroyAllWindows()
    except Exception as ex:
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print message
    heatMap_Generator(path_to_backsub, path_to_heatmap).run()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
