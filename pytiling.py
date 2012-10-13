#! /usr/bin/python

from __future__ import division
from argparse import ArgumentParser
import numpy as np
import Image, os, random, heapq

def getsources(dir, num_sources, aspect_ratio, thumb_width):
    """
    Get the source images and process them to fit into the tiles
    of the target image and extract their average color
    """
    files = random.sample(os.listdir(dir), num_sources)
    sources = []
    for f in files:
        src = Image.open(os.path.join(dir, f))
        src_x, src_y = src.size
        width = min(src_x, int(src_y*aspect_ratio))
        height = min(int(src_x/aspect_ratio), src_y)
        left = int((src_x - width)/2)
        top = int((src_y - height)/2)
        box = (left, top, left+width, top+height)
        src = src.crop(box)
        src = src.resize((thumb_width, int(thumb_width/aspect_ratio)), Image.ANTIALIAS)
        src_array = np.asarray(src)
        sources.append({'image':src, 'mean':np.mean(np.mean(src_array,0),0)/255})
    return sources

def scansources(sources, t_mean):
    """
    Scan the source arrays and compare them with the target array
    by their square euclidean distance. The results are inserted
    into a heap sorted by their similarity score.
    """
    t_mean_norm = t_mean/255
    heap = []
    for s in sources:
        score = ((s['mean'] - t_mean_norm)**2).sum()
        heapq.heappush(heap, (score, s))
    return heapq.heappop(heap)[1]['image']

def mosaicify(target, sources_dir, num_sources=50, num_tiles=72, max_width=800):
    num_tiles = num_tiles
    num_sources = num_sources
    max_width = max_width

    # open the target image
    _target = Image.open(target)

    # store its dimensions
    _width, _height = _target.size

    # get the ratio between its dimensions
    ratio = _width/_height

    # get the source images. crop and rescale them.
    sources = getsources(sources_dir, num_sources, ratio, int(max_width/num_tiles))

    # rescale the image so that it has one pixel for each tile
    _target = _target.resize((num_tiles, num_tiles), Image.ANTIALIAS)

    # store its new dimensions
    pixwidth, pixheight = _target.size

    # scan the source images for matches with the target tiles
    # and composite a new image made of the selected tiles
    _target = np.asarray(_target)
    # a new image to hold the tiles
    composite = Image.new('RGB', (sources[0]['image'].size[0]*num_tiles, sources[0]['image'].size[1]*num_tiles))
    # composite the final image
    for h in range(pixheight):
        for w in range(pixwidth):
            res = scansources(sources, _target[h][w])
            composite.paste(res, (res.size[0]*w, res.size[1]*h))

    return composite

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--target', help='The target image path.')
    parser.add_argument('-s', '--sources', help='The source images directory.')
    parser.add_argument('-n', '--num_sources', help='The number of random source images to use for tiling.', default=50)
    parser.add_argument('-l', '--num_tiles', help='The number of tiles the final image will have (in both dimensions).', default=72)
    parser.add_argument('-w', '--max_width', help='The width of the final image in pixels.', default=800)
    args = parser.parse_args()
    mosaicify(args.target, args.sources, args.num_sources, args.num_tiles, args.max_width)