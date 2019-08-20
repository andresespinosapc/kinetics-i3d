import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, functional as TF
import tensorflow as tf
from collections import deque

import i3d


parser = argparse.ArgumentParser()
parser.add_argument('frames_dir')
parser.add_argument('--checkpoint_path', default='data/checkpoints/rgb_imagenet/model.ckpt')
parser.add_argument('--n_context_frames', type=int, default=6)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--out_names_file', default='video_names.pkl')
parser.add_argument('--out_feats_file', default='video_feats.npy')
args = parser.parse_args()

window_size = args.n_context_frames * 2 + 1

# CHECKPOINT_PATH = 'data/checkpoints/rgb_imagenet/model.ckpt'
# FRAMES_DIR = '/home/andres/workspace/something-to-something'
# N_CONTEXT_FRAMES = 1
# window_size = N_CONTEXT_FRAMES * 2 + 1
# IMAGE_SIZE = 224
# BATCH_SIZE = 4
# VERBOSE = False


video_names = os.listdir(args.frames_dir)

class RescalePixels:
    def __call__(self, img):
        return np.subtract(np.divide(img, 127.5), 1)
        # return img / 127.5 - 1

transform = Compose([
    Resize(256),
    CenterCrop(args.image_size),
    RescalePixels(),
])

# DEFINE MODEL
rgb_input = tf.placeholder(
    tf.float32,
    shape=(None, window_size, args.image_size, args.image_size, 3)
)
with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(
        400, spatial_squeeze=True, final_endpoint='Mixed_4f'
    )
    rgb_feats, _ = rgb_model(
        rgb_input, is_training=False, dropout_keep_prob=1.0
    )
rgb_variable_map = {}
eval_type = 'rgb'
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
            rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
            rgb_variable_map[variable.name.replace(':0', '')] = variable
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

with tf.Session() as sess:
    feed_dict = {}
    rgb_saver.restore(sess, args.checkpoint_path)
    tf.logging.info('RGB checkpoint restored')

    # LOAD DATA
    final_feats = []
    queue = np.empty((args.batch_size, window_size, args.image_size, args.image_size, 3))
    queue_pointer = 0
    queue_sizes = deque()
    pending = False
    pbar = tqdm(list(enumerate(video_names)))
    for i_video, video_name in pbar:
        video_path = os.path.join(args.frames_dir, video_name)
        frame_names = os.listdir(video_path)
        frames = np.empty((len(frame_names), args.image_size, args.image_size, 3))
        for i, frame_name in enumerate(frame_names):
            img = Image.open(os.path.join(video_path, frame_name))
            transformed_img = transform(img)
            frames[i] = transformed_img

        video_batch_size = frames.shape[0] - window_size + 1
        if video_batch_size <= 0:
            frames = np.pad(frames, ((0, -video_batch_size+1,), (0, 0), (0, 0), (0, 0)), 'constant')
            video_batch_size = 1
        queue_sizes.append(video_batch_size)

        for i in range(video_batch_size):
            final_round = (i_video == len(video_names)-1) and (i == video_batch_size-1)
            if final_round:
                args.batch_size = queue_pointer + 1
            if args.verbose:
                print('Adding to queue at {}. Frame {}:{} of video {}'.format(queue_pointer, i, i+window_size, video_name))
            queue[queue_pointer] = frames[i:i+window_size]
            queue_pointer = (queue_pointer + 1) % args.batch_size
            
            if queue_pointer == 0:
                if args.verbose:
                    print('Running model')
                # Run model
                feed_dict[rgb_input] = queue

                out_feats, = sess.run(
                    [rgb_feats],
                    feed_dict=feed_dict)

                # Save values into final_feats
                pointer = 0
                if pending:
                    queue_size = queue_sizes.popleft()
                    if queue_size <= args.batch_size:
                        pending = False
                        pointer = queue_size
                    else:
                        queue_sizes.appendleft(queue_size - args.batch_size)
                        queue_size = args.batch_size
                    if args.verbose:
                        print('Appending feats {}:{} to last one'.format(0, queue_size))
                    final_feats[-1] = np.concatenate([final_feats[-1], out_feats[0:queue_size]])
                if not pending:
                    while pointer < args.batch_size:
                        queue_size = queue_sizes.popleft()
                        new_pointer = pointer + queue_size
                        if new_pointer > args.batch_size:
                            pending = True
                            queue_sizes.appendleft(new_pointer - args.batch_size)
                            new_pointer = args.batch_size
                        if args.verbose:
                            print('Appending feats {}:{}'.format(pointer, new_pointer))
                        final_feats.append(out_feats[pointer:new_pointer])
                        pointer = new_pointer
    print('Saving files...')
    pickle.dump(video_names, open(args.out_names_file, 'wb'))
    np.save(args.out_feats_file, final_feats)
