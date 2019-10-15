import os
import argparse

def convert(fpath):

    assert fpath.endswith('h264'), 'Source file must have .h264 file name extension.'

    target_fpath = fpath[:-5] + '.avi'

    # command = 'ffmpeg -i ' + fpath + \
    #           ' -pix_fmt yuv420p ' + \
    #           '-vf scale=w=-1:h=-1:in_range=pc:out_range=tv ' + \
    #           '-c:v libx264 ' + target_fpath

    command = 'ffmpeg -i ' + fpath + \
              ' -pix_fmt yuv420p ' + \
              '-vf eq=gamma=1.5:saturation=1.5 ' + \
              '-c:v libx264 ' + target_fpath

    os.system(command)

# Running the following command on the output .avi file yields good luminance quality
# ffmpeg -i original.avi -vf eq=gamma=1.5:saturation=1.5 -c:a copy  outfile.avi


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Convert h264 to mjpeg, conserving full color range.')
    parser.add_argument('source_path', type=str, nargs=1, 
                        help='Path to H264 encoded video.')
    args = parser.parse_args()
    convert(args.source_path[0])
