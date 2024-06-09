import os

SAMPLE_VIDEO_DIRECTORY = os.path.join(os.path.abspath(''), 'data', '171204_pose1_sample', 'hdVideos')
VID_FILE_NAME_BASE = 'hd_00_'
OUTPUT_FOLDER_BASE = os.path.join(os.path.abspath(''), 'data_processed', 'sample', 'vid_output', 'vid')
OUTPUT_FOLDER_COLMAP = os.path.join(os.path.abspath(''), 'data_processed', 'sample', 'colmap_output', 'colmap')
OUTPUT_FOLDER_MODEL = os.path.join(os.path.abspath(''), 'data_processed', 'sample', 'model_output', 'frame')
SPLATS_OUTPUT = os.path.join(os.path.abspath(''), 'data_processed', 'sample', 'gaussians', 'frame')
FRAME_NAME_BASE = 'frame_'
FRAME_NUM = 101
VIDEO_NUM = 30

for i in range(30):
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

for j in range(FRAME_NUM):
    current_output = SPLATS_OUTPUT + '_' + str(j + 1)
    current_input = os.path.join(OUTPUT_FOLDER_MODEL, 'colmap_' + str(j + 1), 'splatfacto')
    for dateDir in os.listdir(current_input):
        os.system('ns-export gaussian-splat --load-config ' + os.path.join(current_input, dateDir, 'config.yml')
                  + ' --output-dir ' + current_output)
