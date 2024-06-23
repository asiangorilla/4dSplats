import os, argparse

SAMPLE_VIDEO_DIRECTORY = os.path.join(os.path.abspath(''), 'data', '171204_pose1_sample', 'hdVideos')
VID_FILE_NAME_BASE = 'hd_00_'
OUTPUT_FOLDER = os.path.join(os.path.abspath(''), 'data_processed')
FRAME_NAME_BASE = 'frame_'
FRAME_NUM = 101
VIDEO_NUM = 30  # is it even needed?
IMAGE_FILES = ['images', 'images_2', 'images_4', 'images_8']

help_msg = 'some kind of message'

parser = argparse.ArgumentParser(description=help_msg)

parser.add_argument('-o', '--output_path', type=str, default=OUTPUT_FOLDER,
                    help='Specifies output folder path', dest='output_path')
parser.add_argument('-i', '--input_path', type=str, default=SAMPLE_VIDEO_DIRECTORY,
                    help='Specifies input folder path', dest='input_path')

args = parser.parse_args()

'''
for i in range(VIDEO_NUM):
    if i < 10:
        print('ns-process-data video --skip-colmap '
              '--data ' + os.path.join(SAMPLE_VIDEO_DIRECTORY, VID_FILE_NAME_BASE + '0' + str(i) + '.mp4') +
              ' --output-dir ' + OUTPUT_FOLDER_BASE + str(i))
        os.system('ns-process-data video --skip-colmap '
                  '--data ' + os.path.join(SAMPLE_VIDEO_DIRECTORY, VID_FILE_NAME_BASE + '0' + str(i) + '.mp4') +
                  ' --output-dir ' + OUTPUT_FOLDER_BASE + str(i))
    else:
        print('ns-process-data video --skip-colmap '
              '--data ' + os.path.join(SAMPLE_VIDEO_DIRECTORY, VID_FILE_NAME_BASE + str(i) + '.mp4') +
              ' --output-dir ' + OUTPUT_FOLDER_BASE + str(i))
        os.system('ns-process-data video --skip-colmap '
                  '--data ' + os.path.join(SAMPLE_VIDEO_DIRECTORY, VID_FILE_NAME_BASE + str(i) + '.mp4') +
                  ' --output-dir ' + OUTPUT_FOLDER_BASE + str(i))
'''

OUTPUT_FOLDER_BASE = os.path.join(os.path.abspath(args.output_path), 'vid_output')
OUTPUT_FOLDER_ORDERED = os.path.join(os.path.abspath(args.output_path), 'ordered_output')
OUTPUT_FOLDER_COLMAP = os.path.join(os.path.abspath(args.output_path), 'colmap_output')
OUTPUT_FOLDER_MODEL = os.path.join(os.path.abspath(args.output_path), 'model_output')
SPLATS_OUTPUT = os.path.join(os.path.abspath(args.output_path), 'gaussians', 'frame')

directory_inp = os.fsencode(os.path.abspath(args.input_path))

for ind, file in enumerate(os.listdir(directory_inp)):
    filename = os.fsdecode(file)
    if filename.endswith('.mp4'):
        print('ns-process-data video --skip-colmap '
              '--data ' + os.path.join(args.input_path, filename) +
              ' --output-dir ' + OUTPUT_FOLDER_BASE + 'vid' + str(ind))
        os.system('ns-process-data video --skip-colmap '
                  '--data ' + os.path.join(args.input_path, filename) +
                  ' --output-dir ' + os.path.join(OUTPUT_FOLDER_BASE, 'vid' + str(ind)))

'''
for j in range(FRAME_NUM):
    current_output = ''
    if j + 1 < 10:
        current_output = OUTPUT_FOLDER_BASE + '_frame0000' + str(j + 1)
        print('mkdir ' + current_output)
        os.system('mkdir ' + current_output)
    elif j + 1 < 100:
        current_output = OUTPUT_FOLDER_BASE + '_frame000' + str(j + 1)
        print('mkdir ' + current_output)
        os.system('mkdir ' + current_output)
    else:
        current_output = OUTPUT_FOLDER_BASE + '_frame00' + str(j + 1)
        print('mkdir ' + current_output)
        os.system('mkdir ' + current_output)
    for i in range(VIDEO_NUM):
        if j + 1 < 10:
            h_j = '0000' + str(j + 1)
        elif j + 1 < 100:
            h_j = '000' + str(j + 1)
        else:
            h_j = '00' + str(j + 1)
        print('copy ' + os.path.join(OUTPUT_FOLDER_BASE + str(i), 'images', FRAME_NAME_BASE + h_j + '.png') + ' '
              + os.path.join(current_output, 'frame_from_' + str(i) + '.png'))
        os.system('copy ' + os.path.join(OUTPUT_FOLDER_BASE + str(i), 'images', FRAME_NAME_BASE + h_j + '.png') + ' '
                  + os.path.join(current_output, 'frame_from_' + str(i) + '.png'))
'''

directory_out_base = os.fsencode(OUTPUT_FOLDER_BASE)

for ind, folder in enumerate(os.listdir(directory_out_base)):
    folder_name = os.fsdecode(folder)
    vid_num = folder_name.replace('vid', '')
    for ind_2, image_file in enumerate(os.listdir(folder_name)):
        image_file_name = os.fsdecode(image_file)
        frame_num = image_file_name.split('.')[0].split('_')[1]
        print('copy ' + os.path.join(OUTPUT_FOLDER_BASE + folder_name, IMAGE_FILES[0], image_file_name) + ' '
              + os.path.join(OUTPUT_FOLDER_ORDERED + 'vid_frame' + frame_num, 'frame_from_' + vid_num + '.png'))
        os.system('copy ' + os.path.join(OUTPUT_FOLDER_BASE + folder_name, IMAGE_FILES[0], image_file_name) + ' '
                  + os.path.join(OUTPUT_FOLDER_ORDERED, 'vid_frame' + frame_num, 'frame_from_' + vid_num + '.png'))

'''
for j in range(FRAME_NUM):
    current_input = ''
    if j + 1 < 10:
        current_input = OUTPUT_FOLDER_BASE + '_frame0000' + str(j + 1)
    elif j + 1 < 100:
        current_input = OUTPUT_FOLDER_BASE + '_frame000' + str(j + 1)
    else:
        current_input = OUTPUT_FOLDER_BASE + '_frame00' + str(j + 1)
    print('ns-process-data images --matching-method exhaustive --data ' +
          current_input
          + ' --output-dir '
          + OUTPUT_FOLDER_COLMAP + '_' + str(j + 1))
    os.system('ns-process-data images --matching-method exhaustive --data ' +
              current_input
              + ' --output-dir '
              + OUTPUT_FOLDER_COLMAP + '_' + str(j + 1)
              )
'''

directory_out_ordered = os.fsencode(OUTPUT_FOLDER_ORDERED)

for folder in os.listdir(directory_out_ordered):
    print('ns-process-data images --matching-method exhaustive --data ' +
          OUTPUT_FOLDER_ORDERED + os.fsdecode(folder)
          + ' --output-dir '
          + OUTPUT_FOLDER_COLMAP + 'colmap_' +
          os.fsdecode(folder).split('_')[1].replace('frame', ''))
    os.system('ns-process-data images --matching-method exhaustive --data ' +
              os.path.join(OUTPUT_FOLDER_ORDERED, os.fsdecode(folder))
              + ' --output-dir '
              + OUTPUT_FOLDER_COLMAP + 'colmap_' +
              os.fsdecode(folder).split('_')[1].replace('frame', ''))
'''
for j in range(FRAME_NUM):
    current_input = ''
    if j + 1 < 10:
        current_input = OUTPUT_FOLDER_COLMAP + '_' + str(j + 1)
    elif j + 1 < 100:
        current_input = OUTPUT_FOLDER_COLMAP + '_' + str(j + 1)
    else:
        current_input = OUTPUT_FOLDER_COLMAP + '_' + str(j + 1)
    print('ns-train splatfacto --data ' + OUTPUT_FOLDER_COLMAP + '_' + str(j + 1) + ' --output-dir '
          + OUTPUT_FOLDER_MODEL + ' --save-only-latest-checkpoint False --viewer.quit-on-train-completion True')
    os.system('ns-train splatfacto --data ' + current_input + ' --output-dir ' + OUTPUT_FOLDER_MODEL
              + ' --save-only-latest-checkpoint False --viewer.quit-on-train-completion True')
'''

directory_out_colmap = os.fsencode(OUTPUT_FOLDER_COLMAP)

for folder in os.listdir(directory_out_colmap):
    folder_name = os.fsdecode(folder)
    print('ns-train splatfacto --data ' + os.path.join(OUTPUT_FOLDER_COLMAP, folder_name) + ' --output-dir '
          + OUTPUT_FOLDER_MODEL + ' --save-only-latest-checkpoint False --viewer.quit-on-train-completion True')
    os.system('ns-train splatfacto --data ' + os.path.join(OUTPUT_FOLDER_COLMAP, folder_name) + ' --output-dir '
              + OUTPUT_FOLDER_MODEL + ' --save-only-latest-checkpoint False --viewer.quit-on-train-completion True')

'''
for j in range(FRAME_NUM):
    current_output = SPLATS_OUTPUT + '_' + str(j + 1)
    current_input = os.path.join(OUTPUT_FOLDER_MODEL, 'colmap_' + str(j + 1), 'splatfacto')
    for dateDir in os.listdir(current_input):
        os.system('ns-export gaussian-splat --load-config ' + os.path.join(current_input, dateDir, 'config.yml')
                  + ' --output-dir ' + current_output)
'''

directory_out_model_out = os.fsencode(OUTPUT_FOLDER_MODEL)

for folder in os.listdir(directory_out_model_out):
    folder_name = os.fsdecode(folder)
    for timeDir in os.listdir(os.path.join(OUTPUT_FOLDER_MODEL, folder_name, 'splatfacto')):
        os.system('ns-export gaussian-splat --load-config ' +
                  os.path.join(os.path.join(OUTPUT_FOLDER_MODEL, folder_name, 'splatfacto'),
                               timeDir, 'config.yml')
                  + ' --output-dir ' + SPLATS_OUTPUT + '_' + folder_name.split('_')[1])
