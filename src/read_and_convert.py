# plyfile_read_and_convert.py

from plyfile import PlyData
import pandas as pd
import sys

def read_ply_header(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break
            print(header.decode('ascii'))
    except Exception as e:
        print(f"Error reading PLY file header: {e}")

def convert_ply_to_csv(input_file, output_file):
    try:
        plydata = PlyData.read(input_file)

        # Define the columns to extract
        columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 
                   'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

        # Create a dictionary to store the data
        data = {column: plydata['vertex'].data[column] for column in columns}

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)

        print(f"Filtered data has been saved to {output_file}")
    except Exception as e:
        print(f"Error converting PLY to CSV: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plyfile_read_and_convert.py <input_ply_file> <output_csv_file>")
        sys.exit(1)

    input_ply_file = sys.argv[1]
    output_csv_file = sys.argv[2]

    # Read and print the PLY file header
    read_ply_header(input_ply_file)

    # Convert the PLY file to CSV
    convert_ply_to_csv(input_ply_file, output_csv_file)
