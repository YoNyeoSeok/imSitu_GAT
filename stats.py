import argparse
import glob
import json

# def parser(parser=argparse.ArgumentParser()):
#     parser.add_argument('--state', type=str, nargs='+', choices=['verbs', 'images', 'semantic_roles', ])
#     return parser

def json_load(name):
    with open(name) as f:
        j = json.load(f)
    return j
dev, space, train, test = [json_load(name) for name in sorted(glob.glob('./*.json'))]
print(space.keys())
# space = {
#     verbs:
#     nouns:
# }
space_verbs, space_nouns = space.values()

def split2images_situs(split):
    split_images = list(dev.keys())
    split_situs = list(dev.values())
    return split_images, split_situs
def situ2frames_verbs(situ):
    situ_frames = situ['frames']
    situ_verbs = situ['verb']
    return situ_frames, situ_verbs
def frame2roles_nouns(frame):
    frame_roles = list(frame.keys())
    frame_nouns = list(frame.values())
    return frame_roles, frame_nouns

def situs2frames_verbs(situs):
    situs_frame_verb = [situ2frames_verbs(situ) for situ in situs]
    situs_frames, situs_verbs = list(zip(*situs_frame_verb))
    return situs_frames, situs_verbs
def frames2roles_nouns(frames):
    frames_role_noun = [frame2roles_nouns(frame) for frame in frames]
    frames_roles, frames_nouns = list(zip(*frames_role_noun))
    return frames_roles, frames_nouns
def situs_frames2roles_nouns(situs_frames):
    situs_frames_role_noun = [frames2roles_nouns(situ_frames) for situ_frames in situs_frames]
    situs_frames_roles, situs_frames_nouns = list(zip(*situs_frames_role_noun))
    return situs_frames_roles, situs_frames_nouns

def split2frames_verbs(split):
    _, split_situs = split2images_situs(split)
    split_situs_frames, split_situs_verbs = situs2frames_verbs(split_situs)
    return split_situs_frames, split_situs_verbs
def split2roles_nouns(split):
    _, split_situs = split2images_situs(split)
    split_situs_frames, _ = situs2frames_verbs(split_situs)
    split_situs_frames_roles, split_situs_frames_nouns = situs_frames2roles_nouns(split_situs_frames)
    return split_situs_frames_roles, split_situs_frames_nouns

import numpy as np
dev_roles, dev_nouns = split2roles_nouns(dev)
# print(np.unique(dev_roles[0]).tolist())
# print([space_nouns[noun[0]] for noun in dev_nouns[0]])

# dev_situs_frames_roles, dev_situs_frames_nouns = [list(zip(*[list(zip(*roles_nouns(frame))) for frame in situ_frames])) for situ_frames in dev_situs_frames]