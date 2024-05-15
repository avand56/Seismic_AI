import segyio
import glob
import numpy as np
import segyio
import glob
import os

def import_and_save_segy_3d(directory_path, starttime_correction=0):
    """
    Import SEGY files from a directory, apply starttime correction, and save as a 3D .npy array.
    - directory_path: Path where SEGY files are stored and where .npy files will be saved.
    - starttime_correction: Correction to be applied to the start time in the trace header.
    """
    file_pattern = f"{directory_path}/*.sgy"
    files = glob.glob(file_pattern)
    
    for filename in files:
        with segyio.open(filename, "r+", ignore_geometry=True) as f:
            # Apply starttime correction if needed
            if starttime_correction != 0:
                for trace_index in range(f.tracecount):
                    old_start = f.header[trace_index][segyio.TraceField.DelayRecordingTime]
                    f.header[trace_index].update({segyio.TraceField.DelayRecordingTime: old_start + starttime_correction})

            # Extract inline and crossline information
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            num_samples = len(f.samples)

            # Determine the grid dimensions
            min_inline = min(inlines)
            max_inline = max(inlines)
            min_crossline = min(crosslines)
            max_crossline = max(crosslines)

            # Create empty numpy array with correct dimensions
            data = np.zeros((max_inline - min_inline + 1, max_crossline - min_crossline + 1, num_samples))

            # Populate numpy array
            for i, trace in enumerate(f.trace):
                inline_index = inlines[i] - min_inline
                crossline_index = crosslines[i] - min_crossline
                data[inline_index, crossline_index, :] = trace

            # Save the numpy array
            data_filename = os.path.splitext(filename)[0] + '_3D.npy'
            np.save(data_filename, data)
            print(f"Processed and saved {filename} as {data_filename}")

    print(f"All files processed and saved in {directory_path}")

# Example usage
# import_and_save_segy_3d(
#     '/Users/vanderhoeffalex/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Desktop/F3 seismic data plus classification data for machine learning_/Penobscot_3D_gathers_part1'
# )
import segyio
import glob
import numpy as np
import os

def import_and_save_segy_3d_with_metadata(directory_path, starttime_correction=0):
    """
    Import SEGY files from a directory, apply starttime correction, and save as a 3D .npy array
    along with offset information.
    - directory_path: Path where SEGY files are stored and where .npy files will be saved.
    - starttime_correction: Correction to be applied to the start time in the trace header.
    """
    file_pattern = f"{directory_path}/*.sgy"
    files = glob.glob(file_pattern)
    
    for filename in files:
        with segyio.open(filename, "r+", ignore_geometry=True) as f:
            # Apply starttime correction if needed
            if starttime_correction != 0:
                for trace_index in range(f.tracecount):
                    old_start = f.header[trace_index][segyio.TraceField.delay_recording_time]
                    f.header[trace_index][segyio.TraceField.delay_recording_time] = old_start + starttime_correction

            # Extract inline, crossline, and offset information
            inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            offsets = f.attributes(segyio.TraceField.offset)[:]
            num_samples = len(f.samples)

            # Determine the grid dimensions
            min_inline = min(inlines)
            max_inline = max(inlines)
            min_crossline = min(crosslines)
            max_crossline = max(crosslines)
            unique_offsets = np.unique(offsets)

            # Create empty numpy arrays for seismic data and offsets
            seismic_data = np.zeros((len(unique_offsets), max_inline - min_inline + 1, max_crossline - min_crossline + 1, num_samples))
            offset_data = np.zeros((max_inline - min_inline + 1, max_crossline - min_crossline + 1))

            # Populate numpy arrays
            for i, trace in enumerate(f.trace):
                inline_index = inlines[i] - min_inline
                crossline_index = crosslines[i] - min_crossline
                offset_index = np.where(unique_offsets == offsets[i])[0][0]
                seismic_data[offset_index, inline_index, crossline_index, :] = trace
                offset_data[inline_index, crossline_index] = offsets[i]

            # Save the numpy arrays
            seismic_filename = os.path.splitext(filename)[0] + '_3D.npy'
            offset_filename = os.path.splitext(filename)[0] + '_offsets.npy'
            np.save(seismic_filename, seismic_data)
            np.save(offset_filename, offset_data)
            print(f"Processed and saved {filename} as {seismic_filename} and {offset_filename}")

    print(f"All files processed and saved in {directory_path}")

# Example usage
# import_and_save_segy_3d_with_metadata(
#     '/Users/vanderhoeffalex/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Desktop/F3 seismic data plus classification data for machine learning_/Penobscot_3D_gathers_part1'
# )



def import_and_save_segy(directory_path, inline_byte_loc=5, crossline_byte_loc=9, starttime_correction=0):
    """
    Import SEGY files from a directory, specify headers, correct starttime, and save as .npy without assuming regular geometry.
    - directory_path: Path where SEGY files are stored and where .npy files will be saved.
    - inline_byte_loc: Byte location for inline number in SEGY file header
    - crossline_byte_loc: Byte location for crossline number in SEGY file header
    - starttime_correction: Start time correction value
    """
    file_pattern = f"{directory_path}/*.sgy"
    files = glob.glob(file_pattern)
    
    for filename in files:
        with segyio.open(filename, "r+", ignore_geometry=True) as f:
            # Set new headers based on parameters
            for trace_index in range(f.tracecount):
                inline_value = f.header[trace_index][inline_byte_loc]
                crossline_value = f.header[trace_index][crossline_byte_loc]
                f.header[trace_index].update({
                    segyio.TraceField.INLINE_3D: inline_value,
                    segyio.TraceField.CROSSLINE_3D: crossline_value,
                    segyio.TraceField.DelayRecordingTime: starttime_correction
                })
            
            # Collect all traces into a list, then convert to numpy array
            traces = [f.trace[trid] for trid in range(f.tracecount)]
            data = np.stack(traces)
            data_filename = os.path.splitext(filename)[0] + '.npy'
            
            # Save the seismic data as a numpy array
            np.save(data_filename, data)
            
            print(f"Processed and saved {filename} as {data_filename}")

    print(f"All files processed and saved in {directory_path}")

# Example usage
# import_and_save_segy(
#     '/Users/vanderhoeffalex/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Desktop/F3 seismic data plus classification data for machine learning_/Penobscot_3D_gathers_part1',
#     inline_byte_loc=5, 
#     crossline_byte_loc=9, 
#     starttime_correction=0
# )




def segy_to_npy_with_coordinates(segy_file_path, npy_file_path, npy_inline_path, npy_crossline_path):
    """
    Convert a SEG-Y file to a NumPy (.npy) file and save inline and crossline numbers.

    Args:
    segy_file_path (str): Path to the input SEG-Y file.
    npy_file_path (str): Path to the output NPY file for seismic data.
    npy_inline_path (str): Path to the output NPY file for inline numbers.
    npy_crossline_path (str): Path to the output NPY file for crossline numbers.
    """
    # Open the SEG-Y file
    with segyio.open(segy_file_path, "r") as segyfile:
        segyfile.mmap()  # Memory-map the file for better performance
        
        # Extract data as a 3D NumPy array
        data = segyio.tools.cube(segyfile)

        # Extract inline and crossline numbers
        inlines = segyio.tools.collect(segyfile.field(segyio.TraceField.INLINE_3D))
        crosslines = segyio.tools.collect(segyfile.field(segyio.TraceField.CROSSLINE_3D))
    
    # Save the seismic data to a NPY file
    np.save(npy_file_path, data)
    print(f"Seismic data saved to {npy_file_path}")
    
    # Save the inline and crossline numbers to NPY files
    np.save(npy_inline_path, inlines)
    np.save(npy_crossline_path, crosslines)
    print(f"Inline numbers saved to {npy_inline_path}")
    print(f"Crossline numbers saved to {npy_crossline_path}")

# # Example usage
# segy_file_path = 'path_to_your_segy_file.segy'
# npy_file_path = 'path_to_your_output_file.npy'
# npy_inline_path = 'path_to_your_output_inline_file.npy'
# npy_crossline_path = 'path_to_your_output_crossline_file.npy'
# segy_to_npy_with_coordinates(segy_file_path, npy_file_path, npy_inline_path, npy_crossline_path)

def segy_to_npy_prestack(segy_file_path, npy_file_path, npy_inline_path, npy_crossline_path, npy_azimuth_path, npy_offset_path):
    """
    Convert a SEG-Y file to NumPy (.npy) format and save additional pre-stack data characteristics.

    Args:
    segy_file_path (str): Path to the input SEG-Y file.
    npy_file_path (str): Path to the output NPY file for seismic data.
    npy_inline_path (str): Path to the output NPY file for inline numbers.
    npy_crossline_path (str): Path to the output NPY file for crossline numbers.
    npy_azimuth_path (str): Path to the output NPY file for azimuth data.
    npy_offset_path (str): Path to the output NPY file for offset data.
    """
    # Open the SEG-Y file
    with segyio.open(segy_file_path, "r") as segyfile:
        segyfile.mmap()  # Memory-map the file for better performance
        
        # Extract data as a 3D NumPy array
        data = segyio.tools.cube(segyfile)
        
        # Extract inline and crossline numbers
        inlines = np.array([tr.header[segyio.TraceField.INLINE_3D] for tr in segyfile.traces])
        crosslines = np.array([tr.header[segyio.TraceField.CROSSLINE_3D] for tr in segyfile.traces])

        # Extract azimuth and offset
        azimuths = np.array([tr.header[segyio.TraceField.Azimuth] for tr in segyfile.traces])
        offsets = np.array([tr.header[segyio.TraceField.Offset] for tr in segyfile.traces])

    # Save the seismic data and header information to NPY files
    np.save(npy_file_path, data)
    np.save(npy_inline_path, inlines)
    np.save(npy_crossline_path, crosslines)
    np.save(npy_azimuth_path, azimuths)
    np.save(npy_offset_path, offsets)
    
    print(f"Data and trace header information saved:\nData: {npy_file_path}\nInlines: {npy_inline_path}\nCrosslines: {npy_crossline_path}\nAzimuths: {npy_azimuth_path}\nOffsets: {npy_offset_path}")

# # Example usage
# segy_file_path = 'path_to_your_segy_file.segy'
# npy_file_path = 'path_to_your_output_file.npy'
# npy_inline_path = 'path_to_your_output_inline_file.npy'
# npy_crossline_path = 'path_to_your_output_crossline_file.npy'
# npy_azimuth_path = 'path_to_your_output_azimuth_file.npy'
# npy_offset_path = 'path_to_your_output_offset_file.npy'
# segy_to_npy_prestack(segy_file_path, npy_file_path, npy_inline_path, npy_crossline_path, npy_azimuth_path, npy_offset_path)

def segy_to_npy_with_manual_headers(segy_file_path, npy_file_path):
    with segyio.open(segy_file_path, "r", ignore_geometry=True) as segyfile:
        segyfile.mmap()
        # Get total number of traces
        num_traces = segyfile.tracecount
        
        # Initialize arrays for inlines and crosslines
        inlines = np.zeros(num_traces, dtype=int)
        crosslines = np.zeros(num_traces, dtype=int)

        # Read each trace header
        for i in range(num_traces):
            inlines[i] = segyfile.header[i][segyio.TraceField.INLINE_3D]
            crosslines[i] = segyfile.header[i][segyio.TraceField.CROSSLINE_3D]

        # Check for duplicates or inconsistencies in crosslines
        unique_crosslines = np.unique(crosslines)
        if len(unique_crosslines) != len(crosslines):
            print("Warning: Non-unique crossline numbers found.")

        # Assuming regular sampling, reshape data into a cube
        max_inline = max(inlines)
        max_crossline = max(crosslines)
        data = np.zeros((max_inline + 1, segyfile.samples.size, max_crossline + 1))

        for i, trace in enumerate(segyfile.trace):
            il = inlines[i]
            cl = crosslines[i]
            data[il, :, cl] = trace

    np.save(npy_file_path, data)
    print(f"Data saved to {npy_file_path}")

# Use this to manually handle and validate headers
# segy_to_npy_with_manual_headers(segy_file_path, npy_file_path)