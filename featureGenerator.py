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
from fnmatch import fnmatch

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=2000,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7,
                      useHarrisDetector=0,
                      k=0.04
                      )

'''
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
'''


class App:

    def __init__(self, video_src, mask, label, path_to_frames):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.points = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.color = np.random.randint(0, 255, (100, 3))
        self.vectors = []
        self.mask = cv2.imread(mask, 0)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.label = label
        self.path_to_frames = path_to_frames

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

    def tryint(self, s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(self, s):
        return [self.tryint(c) for c in re.split('([0-9]+)', s)]

    def run(self):
        pos_frame = self.cam.get(1)
        frame_count = 0
        files = []
        for name in sorted(os.listdir(self.path_to_frames)):
            if fnmatch(name, "*.png"):
                #print(os.path.join(self.path_to_frames, name))
                files.append(str(os.path.join(self.path_to_frames, name)))
        files.sort(key=self.alphanum_key)
        numOfFiles = len(files)
        timePassed = 0
        numCubes = 0
        fileCount = 0
        features = []
        '''
        Remove te condition for 10 or 20 frames, it's only only for testing purposed
        '''
        while True and frame_count < 20:
            ret, frame = self.cam.read()
            '''print('One Loop')
            print(frame_count)
            print(files[frame_count])'''
            if frame_count == fileCount:
                del features[:]
                timePassed += 1
                img_set = []
                img1 = cv2.cvtColor(cv2.imread(
                    files[fileCount]), cv2.COLOR_BGR2GRAY)
                fileCount += 1
                img2 = cv2.cvtColor(cv2.imread(
                    files[fileCount]), cv2.COLOR_BGR2GRAY)
                fileCount += 1
                img3 = cv2.cvtColor(cv2.imread(
                    files[fileCount]), cv2.COLOR_BGR2GRAY)
                fileCount += 1
                img4 = cv2.cvtColor(cv2.imread(
                    files[fileCount]), cv2.COLOR_BGR2GRAY)
                fileCount += 1
                img5 = cv2.cvtColor(cv2.imread(
                    files[fileCount]), cv2.COLOR_BGR2GRAY)
                fileCount += 1
                img_set.extend((img1, img2, img3, img4, img5))
                # print(timePassed)
                resize_2020_image_set = []
                resize_4030_image_set = []
                resize_160120_image_set = []
                for image in img_set:
                    resize_2020_image_set.append(cv2.resize(image, (20, 20)))
                    resize_4030_image_set.append(cv2.resize(image, (40, 30)))
                    resize_160120_image_set.append(
                        cv2.resize(image, (160, 120)))
                resized_image_set = [resize_2020_image_set,
                                     resize_4030_image_set, resize_160120_image_set]
                patches_all = [[], [], []]
                iterator = 0
                for images_set in resized_image_set:
                    for resized_img in images_set:
                        patch_list = []
                        patch = []
                        for start in range(0, len(resized_img[0]), 10):
                            count = 1
                            for row in resized_img:
                                patch.append(row[start:start + 10])
                                if(count == 10):
                                    count = 0
                                    patch_list.append(patch)
                                    patch = []
                                count += 1
                        patches_all[iterator].append(patch_list)
                    iterator += 1

                cubes = []

                for resolution_patch_set in patches_all:
                    for iterator in range(len(resolution_patch_set[0])):
                        p_one = resolution_patch_set[0][iterator]
                        p_two = resolution_patch_set[1][iterator]
                        p_three = resolution_patch_set[2][iterator]
                        p_four = resolution_patch_set[3][iterator]
                        p_five = resolution_patch_set[4][iterator]
                        cubes.append([p_one, p_two, p_three, p_four, p_five])
                numCubes += len(cubes)
                for cub in cubes:
                    sobelx = cv2.Sobel(
                        np.array(cub), cv2.CV_64F, 1, 0, ksize=-1)
                    sobely = cv2.Sobel(
                        np.array(cub), cv2.CV_64F, 0, 1, ksize=-1)
                    sobelt = cv2.Sobel(np.array(zip(*cub)),
                                       cv2.CV_64F, 0, 1, ksize=-1)
                    sobelt = zip(*sobelt)
                    feature = []
                    for time_value in range(5):
                        for y_value in range(10):
                            for x_value in range(10):
                                feature.append(sobelx[time_value][
                                               y_value][x_value])
                                feature.append(sobely[time_value][
                                               y_value][x_value])
                                feature.append(sobelt[time_value][
                                               y_value][x_value])
                    features.append(feature)
                #features = np.array(features)
                # print(features.shape)

            if ret:
                frame_vectors = []
                frame = cv2.fastNlMeansDenoisingColored(
                    frame, None, 10, 10, 7, 21)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
                pos_frame = self.cam.get(1)
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
                                1], dst, point.angle, point.response]
                            while(len(vtr) < 1500):
                                vtr.append(0)
                            # self.vectors.append(vtr)
                            frame_vectors.append(vtr)
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

                if self.frame_idx % self.detect_interval == 0:
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

                self.frame_idx += 1
                self.prev_gray = frame_gray
                # print(len(frame_vectors))
                zeroes = []
                while(len(zeroes) < 1500):
                    zeroes.append(0)
                while(len(frame_vectors) < 2000):
                    frame_vectors.append(zeroes)
                while(len(features) < 2000):
                    features.append(zeroes)
                # print(len(frame_vectors))
                # frame_vectors=np.array(frame_vectors)
                # print(frame_vectors.shape)
                # featureVal=np.array(features)
                # print(featureVal.shape)
                self.vectors.append(
                    tuple((frame_vectors, features, self.label)))
                cv2.imshow('Frames', vis)
                frame_count += 1
                #print('Tracking frame ' + str(frame_count))
            else:
                self.cam.set(1, pos_frame - 1)
                cv2.waitKey(1000)

            ch = cv2.waitKey(1)
            if ch == 27 or self.cam.get(1) == self.cam.get(7):
                break
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
            cv2.imwrite(os.path.join(path_to_frames, str(
                frame_count) + '.png'), frame, [int(cv2.IMWRITE_PNG_STRATEGY_DEFAULT)])
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
    '''
    Uncomment for extracting frames from video
    '''
    videoToFrames(video_src, path_to_frames)
    '''
        Code for creating mask by background substraction and and motion heatmap
    '''
    maskGenerator.main(path_to_frames, path_to_backsub, path_to_heatmap)
    try:
        first_file = next(os.path.join(path_to_heatmap, f) for f in os.listdir(
            path_to_heatmap) if os.path.isfile(os.path.join(path_to_heatmap, f)))
        features = App(video_src, first_file, label, path_to_frames).run()
        featureList = pandas.DataFrame(features)
        featureList.columns = ['features', 'pixels', 'labels']
        #featureList=featureList[featureList['features'].map(len) > 0]
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
