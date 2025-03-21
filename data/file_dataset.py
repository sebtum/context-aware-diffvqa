# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import os
import torch
import pickle


class FileDataset:
    def __init__(self, file_path, selected_col_ids=None, dtypes=None, separator="\t", cached_index=True):
        self.file_path = file_path
        assert os.path.exists(self.file_path), "Error: The local datafile {} not exists!".format(self.file_path)

        self.separator = separator
        if selected_col_ids is None:
            # default to all fields
            self.selected_col_ids = list(
                range(len(open(self.file_path).readline().rstrip("\n").split(self.separator))))
        else:
            self.selected_col_ids = [int(col_id) for col_id in selected_col_ids.split(",")]
        if dtypes is None:
            # default to str
            self.dtypes = [str for col_id in self.selected_col_ids]
        else:
            self.dtypes = [eval(col_dtype) for col_dtype in dtypes.split(",")]
            assert len(self.dtypes) == len(self.selected_col_ids)

        self.data_cnt = 0
        try:
            self.slice_id = torch.distributed.get_rank()
            self.slice_count = torch.distributed.get_world_size()
        except Exception:
            self.slice_id = 0
            self.slice_count = 1
        self.cached_index = cached_index
        self._init_seek_index()
        #self._reader = self._get_reader() ####
        print("file {} slice_id {} row count {} total row count {}".format(
            self.file_path, self.slice_id, self.row_count, self.total_row_count)
        )

    def _init_seek_index(self):
        # Build a cache file name using the base name of the input file.
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        cache_path = os.path.join(os.path.dirname(self.file_path), base_name + "_index.pkl")
        if self.slice_id == 0:
            # Rank 0: create or load the cache file.
            if self.cached_index and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.total_row_count, self.lineid_to_offset = pickle.load(f)
                print(f"local datafile {self.file_path} rank {self.slice_id} using cached row_count and line_idx-to-offset mapping from {cache_path}")
            else:
                with open(self.file_path, "rb") as fp:
                    print(f"local datafile {self.file_path} rank {self.slice_id} begin to initialize row_count and line_idx-to-offset mapping")
                    # Skip header row
                    fp.readline()
                    offset = fp.tell()
                    self.total_row_count = 0
                    self.lineid_to_offset = []
                    for line in fp:
                        self.lineid_to_offset.append(offset)
                        self.total_row_count += 1
                        offset = fp.tell()
                if self.cached_index:
                    with open(cache_path, "wb") as f:
                        pickle.dump((self.total_row_count, self.lineid_to_offset), f)
                    print(f"local datafile {self.file_path} rank {self.slice_id} created cache file at {cache_path}")
        # Synchronize all processes
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # All ranks load the cache file (even rank 0 can reload if desired)
        if self.slice_id != 0:
            with open(cache_path, "rb") as f:
                self.total_row_count, self.lineid_to_offset = pickle.load(f)
            print(f"local datafile {self.file_path} rank {self.slice_id} loaded cache file from {cache_path}")
#        # If caching is enabled and the cache file exists, load it.
#        if self.cached_index and os.path.exists(cache_path):
#            with open(cache_path, "rb") as f:
#                self.total_row_count, self.lineid_to_offset = pickle.load(f)
#            print("local datafile {} slice_id {} using cached row_count and line_idx-to-offset mapping from {}".format(
#                self.file_path, self.slice_id, cache_path))
#        else:
#            # make an iteration over the file to get row_count and line_idx-to-offset mapping
#            with open(self.file_path, "rb") as fp:
##            with open(self.file_path, "r") as fp:
#                print("local datafile {} slice_id {} begin to initialize row_count and line_idx-to-offset mapping".format(
#                    self.file_path, self.slice_id))
#                # Skip the header row
#                fp.readline()
#                # Initialize row count and offsets
#                offset = fp.tell()  # Start tracking after the header
#                self.total_row_count = 0
#                self.lineid_to_offset = []
#
#                for line in fp:
#                    #print(f"line {self.total_row_count}: offset {offset}")
#                    self.lineid_to_offset.append(offset)
#                    self.total_row_count += 1
#                    offset = fp.tell()
#            # If caching is enabled, save the computed mapping to the cache file.
#            if self.cached_index:
#                with open(cache_path, "wb") as f:
#                    pickle.dump((self.total_row_count, self.lineid_to_offset), f)
#                print("local datafile {} slice_id {} created cache file at {}".format(
#                    self.file_path, self.slice_id, cache_path))
                
        self._compute_start_pos_and_row_count()
        print("local datafile {} slice_id {} finished initializing row_count and line_idx-to-offset mapping".format(
            self.file_path, self.slice_id))

    def _compute_start_pos_and_row_count(self):
        base_row_count = self.total_row_count // self.slice_count
        extra_rows = self.total_row_count % self.slice_count

        # Distribute extra rows to the first `extra_rows` slices
        if self.slice_id < extra_rows:
            self.row_count = base_row_count + 1
            self.start_pos = self.slice_id * self.row_count
        else:
            self.row_count = base_row_count
            self.start_pos = self.slice_id * self.row_count + extra_rows
        
        # Ensure start_pos is within bounds
        if self.start_pos >= len(self.lineid_to_offset):
            raise IndexError(f"start_pos {self.start_pos} is out of range for lineid_to_offset with size {len(self.lineid_to_offset)}")

    def _get_reader(self):
#        fp = open(self.file_path, "r")
        fp = open(self.file_path, "rb")
        fp.seek(self.lineid_to_offset[self.start_pos])
        return fp

    def _seek(self, offset=0):
        # Lazily initialize _reader if necessary
        if not hasattr(self, '_reader') or self._reader.closed:
            self._reader = self._get_reader()
        try:
            target_index = self.start_pos + offset
            if target_index >= len(self.lineid_to_offset) or target_index < 0:
                raise IndexError(f"Seek target index {target_index} is out of range.")
            print("slice_id {} seek offset {}".format(self.slice_id, target_index))
            #traceback.print_stack(limit=10)
            self._reader.seek(self.lineid_to_offset[self.start_pos + offset])
            self.data_cnt = offset
        except Exception:
            print("slice_id {} seek offset {}".format(self.slice_id, target_index))
            self._reader.seek(self.lineid_to_offset[target_index])
            self.data_cnt = offset

    def __del__(self):
        if hasattr(self, "_reader") and self._reader:
            try:
                self._reader.close()
            except Exception as e:
                print(f"Error closing reader: {e}")


    def __len__(self):
        return self.row_count

    def get_total_row_count(self):
        return self.total_row_count
    
    def __getitem__(self, index):
        # Lazy file handle initialization: if not present or closed, open a new reader
        if not hasattr(self, '_reader') or self._reader.closed:
            self._reader = self._get_reader()
            print(f"get reader, start postion {self.start_pos}, index {index}")

        # Compute the absolute index within the full file using your per-slice start position.
        absolute_index = self.start_pos + index
        if absolute_index >= len(self.lineid_to_offset):
            raise IndexError("Index out of range for the file.")

        # Seek directly to the offset for this absolute index
        self._reader.seek(self.lineid_to_offset[absolute_index])
        line = self._reader.readline()
        # Decode the line from binary to string, then process it
        line = line.decode("utf-8")
        if not line:
            raise IndexError("Reached end of file unexpectedly.")
        #else:
        #    print(f"absolute index: {absolute_index}, index {index}, offset: {self.lineid_to_offset[absolute_index]}\n"
        #          f"line until the 60th character: {line[:60]}"
        #          )

        # Split the line and validate that it contains enough columns
        column_l = line.rstrip("\n").split(self.separator)
        if max(self.selected_col_ids) >= len(column_l):
            raise IndexError(f"Obtained columns lenght is out of range for the line:{len(column_l)}, expecting: {self.selected_col_ids}")

        # Convert and return the selected columns
        return [dtype(column_l[col_id]) for col_id, dtype in zip(self.selected_col_ids, self.dtypes)]


#    def __getitem__(self, index):
#        if self.data_cnt == self.row_count:
#            print("reach the end of datafile, start a new reader")
#            self.data_cnt = 0
#            self._reader = self._get_reader()
#        line = self._reader.readline()
#        print(f"data count {self.data_cnt} index {index} line {line[:60]}")
#        column_l = line.rstrip("\n").split(self.separator)
#        self.data_cnt += 1
#        column_l = [dtype(column_l[col_id]) for col_id, dtype in zip(self.selected_col_ids, self.dtypes)]
#        return column_l
