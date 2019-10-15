import cv2
import os
import argparse

class save_video(object):
    def __init__(self, filename, frame_width, frame_height, framerate):
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.out = cv2.VideoWriter(filename, 0x7634706d, framerate, (frame_width,frame_height))

    def write_frame(self, frame):
        self.out.write(frame)

    def close(self):
        self.out.release()

def convert(folder_path, outfile_path, framerate=25):
    '''
    Sorts files by numbers preceding .jpg extension and writes them into an mp4 file.
    '''
    # Get all files in folder
    filenames = os.listdir(folder_path)
    # Sort file names based on numbers
    filenrs = [int(x[:-4]) for x in filenames]
    filenames = [x for _,x in sorted(zip(filenrs,filenames))]
    filenrs = [int(x[:-4]) for x in filenames]
    # Load sample frame
    sampleFrame = cv2.imread(os.path.join(folder_path, filenames[0]))
    frame_height, frame_width, _ = sampleFrame.shape
    # Initialize save_video
    SV = save_video(outfile_path, frame_width, frame_height, framerate)
    # Loop through jpg files and write video
    for n, filename in enumerate(filenames):
        frame = cv2.imread(os.path.join(folder_path, filename))
        SV.write_frame(frame)
        if n % 1000 == 0:
            print(str(n) + ' frames written.')
    # Close video writer
    SV.close()

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Convert all jpg images in folder to mp4 video.')
    parser.add_argument('folder_path', type=str, nargs=1, 
                        help='Path to folder with jpg files.')
    parser.add_argument('outfile_path', type=str, nargs=1, 
                        help='Path of video file.')
    args = parser.parse_args()
    convert(args.folder_path[0], args.outfile_path[0])
