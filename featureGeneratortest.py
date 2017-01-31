#!/usr/bin/env python

'''
Usage- python featureGenerator.py (path_to_source_video) label
label 0 for abnormal, 1 for normal
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import scipy
from scipy import spatial
import cv2
import video
import os
import maskGenerator
import Image
from common import anorm2, draw_str
from time import clock
import re
import csv
import pandas

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=5000,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7,
                      useHarrisDetector=0,
                      k=0.04
                      )


class vector:

    def __init__(self, xCoOrdinates, yCoOrdinates, dst, size, angle, response, octave):
        self.xCoOrdinates = xCoOrdinates
        self.yCoOrdinates = yCoOrdinates
        self.distance = dst
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave

    def getxCoOrdinates(self):
        return self.xCoOrdinates

    def getyCoOrdinates(self):
        return self.yCoOrdinates

    def getDistance(self):
        return self.distance

    def getSize(self):
        return self.size

    def getAngle(self):
        return self.angle

    def getResponse(self):
        return self.response

    def getOctave(self):
        return self.octave


class App:

    def __init__(self, video_src, mask, label):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.points = []
        self.source = video_src
        self.frame_idx = 0
        self.color = np.random.randint(0, 255, (100, 3))
        self.vectors = []
        self.mask = cv2.imread(mask, 0)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.label = label
        #print (self.is_grey_scale(mask))
        # self.mask=cv2.cvtColor(self.mask,cv2.COLOR_BGR2GRAY)
        # cv2.imshow('mask',self.mask)
        # cv2.waitKey(1000)
        #print (self.is_grey_scale(self.mask))
        #self.mask = cv2.convertScaleAbs(self.mask)

    def is_grey_scale(self, img_path):
        im = Image.open(img_path).convert('RGB')
        w, h = im.size
        for i in range(w):
            for j in range(h):
                r, g, b = im.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    def vectorReturn(self):
        return self.vectors

    def run(self):
        frame_count = 0
        '''
        Remove te condition for 20 frames, it's only only for testing purposes

        At least 18 frames needed for motion heatmap
        '''
        for file in os.listdir(self.source):
            if frame_count <= 20:
                frame_count += 1
                # print(frame_count)
                frame = cv2.imread(os.path.join(self.source, file))
                #frame_vectors = []
                frame = cv2.fastNlMeansDenoisingColored(
                    frame, None, 10, 10, 7, 21)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #hist = cv2.calcHist([frame_gray,],[0],None,[256],[0,256])
                vis = frame.copy()
                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1]
                                     for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(
                        img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    for loop, point in zip(p0, self.points):
                        pa, st, err = cv2.calcOpticalFlowPyrLK(
                            img0, img1, loop, None, **lk_params)
                        p0a, st, err = cv2.calcOpticalFlowPyrLK(
                            img1, img0, pa, None, **lk_params)
                        if abs(loop - p0a).reshape(-1, 2).max(-1) < 1:
                            dst = spatial.distance.euclidean(loop, p0a)
                            new_Loop = loop.flatten()
                            vtr = [new_Loop[0], new_Loop[
                                1], dst, point.angle, point.response, frame_count, self.label]
                            # frame_vectors.append(vtr)
                            self.vectors.append(vtr)
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr)
                                        for tr in self.tracks], False, (0, 255, 0))
                    draw_str(vis, (20, 20), 'track count: %d' %
                             len(self.tracks))

                if frame_count % 5 == 0 or frame_count <5:
                    #mask = np.zeros_like(frame_gray)
                    mask = np.zeros_like(self.mask)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    '''p = cv2.goodFeaturesToTrack(
                        frame_gray, mask=self.mask, **feature_params)'''
                    p, des = self.orb.detectAndCompute(frame_gray, self.mask)
                    if p is not None:
                        '''for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])'''
                        for keypoint in p:
                            x = keypoint.pt[0]
                            y = keypoint.pt[1]
                            self.tracks.append([(x, y)])
                            self.points.append(keypoint)

                #self.frame_idx += 1
                self.prev_gray = frame_gray
                #self.vectors.append(tuple((frame_vectors, self.label)))
                cv2.imshow('frame', vis)
                cv2.waitKey(1)
            '''for a in self.vectorReturn():
                print(a.getxCoOrdinates(), a.getyCoOrdinates(), a.getDistance())
            '''
            #print (len(self.tracks))
            # print(len(self.vectors))
        return self.vectorReturn()


def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_basename(path):
    """Extracts basename of a given path. Should Work with any OS Path on any OS"""
    basename = re.search(r'[^\\/]+(?=[\\/]?$)', path)
    if basename:
        return basename.group(0)


def videoToFrames(video_src, path_to_frames):
    cap = video.create_capture(video_src)
    makeDir(path_to_frames)
    frame_count = 0
    pos_frame = cap.get(1)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Frame', frame)
            if frame_count > 0:
                cv2.imwrite(os.path.join(path_to_frames, str(frame_count) + '.png'),
                            frame, [int(cv2.IMWRITE_PNG_STRATEGY_DEFAULT)])
            pos_frame = cap.get(1)
            frame_count += 1
        else:
            cap.set(1, pos_frame - 1)
            cv2.waitKey(500)
        k = cv2.waitKey(1)
        if k == 27 or cap.get(1) == cap.get(7):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    import sys
    try:
        video_src = sys.argv[1]
        label = sys.argv[2]
    except Exception as ex:
        video_src = 0
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    baseName = extract_basename(video_src)
    path_to_backsub = '/home/eshan/Thesis/BackSub/' + baseName
    path_to_heatmap = '/home/eshan/Thesis/Heatmap/' + baseName
    path_to_frames = '/home/eshan/Thesis/Frames/' + baseName
    print(__doc__)
    #videoToFrames(video_src, path_to_frames)
    '''
        Code for creating mask by background substraction and and motion heatmap
    '''
    #maskGenerator.main(path_to_frames, path_to_backsub, path_to_heatmap)
    try:
        first_file = next(os.path.join(path_to_heatmap, f) for f in os.listdir(
            path_to_heatmap) if os.path.isfile(os.path.join(path_to_heatmap, f)))
        features = App(path_to_frames, first_file, label).run()
        featureList = pandas.DataFrame(features)
        featureList.columns = ['x', 'y', 'distance',
                               'angle', 'response', 'frame_no', 'labels']
        #featureList = featureList[featureList['features'].map(len) > 0]
        if not os.path.exists('data.csv'):
            featureList.to_csv('data.csv', index=False)
        else:
            with open('data.csv', 'a') as f:
                featureList.to_csv(f, header=False, index=False)
    except Exception as ex:
        template = "An exception of type {0} occured. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
