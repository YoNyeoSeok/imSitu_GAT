import argparse
import glob
import json
import numpy as np

from imsitu import (imSituVerbRoleNounEncoder,
                    imSituVerbLocalRoleNounEncoder)

# def parser(parser=argparse.ArgumentParser()):
#     parser.add_argument('--state', type=str, nargs='+', choices=['verbs', 'images', 'semantic_roles', ])
#     return parser

def json_load(name):
    with open(name) as f:
        j = json.load(f)
    return j
dev, space, test, train = [json_load(name) for name in sorted(glob.glob('./*.json'))]

train_encoder = imSituVerbRoleNounEncoder(train)
dev_encoder = imSituVerbRoleNounEncoder(dev)
test_encoder = imSituVerbRoleNounEncoder(test)


print(space.keys())
# space = {
#     verbs:
#     nouns:
# }
space_verbs, space_nouns = space.values()

def split2images_annos(split):
    split_images = list(split.keys())
    split_annos = list(split.values())
    return split_images, split_annos
def anno2frames_verbs(anno):
    anno_frames = anno['frames']
    anno_verbs = anno['verb']
    return anno_frames, anno_verbs
def frame2roles_nouns(frame):
    frame_roles = list(frame.keys())
    frame_nouns = list(frame.values())
    return frame_roles, frame_nouns

def annos2frames_verbs(annos):
    annos_frame_verb = [anno2frames_verbs(anno) for anno in annos]
    annos_frames, annos_verbs = list(zip(*annos_frame_verb))
    return annos_frames, annos_verbs
def frames2roles_nouns(frames):
    frames_role_noun = [frame2roles_nouns(frame) for frame in frames]
    frames_roles, frames_nouns = list(zip(*frames_role_noun))
    return frames_roles, frames_nouns
def annos_frames2roles_nouns(annos_frames):
    annos_frames_role_noun = [frames2roles_nouns(anno_frames) for anno_frames in annos_frames]
    annos_frames_roles, annos_frames_nouns = list(zip(*annos_frames_role_noun))
    return annos_frames_roles, annos_frames_nouns

def split2frames_verbs(split):
    _, split_annos = split2images_annos(split)
    split_annos_frames, split_annos_verbs = annos2frames_verbs(split_annos)
    return split_annos_frames, split_annos_verbs
def split2roles_nouns(split):
    _, split_annos = split2images_annos(split)
    split_annos_frames, _ = annos2frames_verbs(split_annos)
    split_annos_frames_roles, split_annos_frames_nouns = annos_frames2roles_nouns(split_annos_frames)
    return split_annos_frames_roles, split_annos_frames_nouns

dev_roles, dev_nouns = split2roles_nouns(dev)
train_roles, train_nouns = split2roles_nouns(train)
test_roles, test_nouns = split2roles_nouns(test)

unique_roles = np.unique(np.concatenate([
    np.unique(np.array(dev_roles)[:, 0]),
    np.unique(np.array(train_roles)[:, 0]),
    np.unique(np.array(test_roles)[:, 0])]))
unique_nouns = np.unique(np.concatenate([
    np.unique(np.concatenate(np.array(train_nouns).flatten())),
    np.unique(np.concatenate(np.array(dev_nouns).flatten())),
    np.unique(np.concatenate(np.array(test_nouns).flatten())),
    ]))
unique_gloss = np.unique(np.concatenate([
    np.unique(np.concatenate([space_nouns[noun]['gloss'] for noun in np.unique(np.concatenate(np.array(train_nouns).flatten()))[1:]])),
    np.unique(np.concatenate([space_nouns[noun]['gloss'] for noun in np.unique(np.concatenate(np.array(dev_nouns).flatten()))[1:]])),
    np.unique(np.concatenate([space_nouns[noun]['gloss'] for noun in np.unique(np.concatenate(np.array(test_nouns).flatten()))[1:]])),
    ]))
# print(np.unique(dev_roles[0]).tolist())
# print([space_nouns[noun[0]] for noun in dev_nouns[0]])

# dev_annos_frames_roles, dev_annos_frames_nouns = [list(zip(*[list(zip(*roles_nouns(frame))) for frame in anno_frames])) for anno_frames in dev_annos_frames]