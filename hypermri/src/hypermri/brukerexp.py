"""
Base class used to read in one Bruker experiment/<expno> directory (see below).

Have a look at the ParaVision Manual for more information on the scan-data folder
structure.

Includes parts of functions copied and adapted from brukerMRI package
(https://github.com/jdoepfert/brukerMRI).
"""

import os
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from brukerapi.dataset import Dataset

from hypermri.utils import init_default_logger, LOG_MODES

# for paper_summary() method only
from IPython.display import display, HTML
import pandas as pd


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


class BrukerExp:
    """Base class for experiments conducted on Bruker preclinical systems.

    ParaVision dataset paths have the following form:

    <DataPath>/<name>/<expno>/pdata/<procno>

    <DataPath>
    This is an arbitrary path in the directory tree that is used as root for
    ParaVision datasets.
    <name>
    This directory contains all information about a study. The name is created by
    ParaVision. The maximum length of the directory name is 64 characters.
    <expno>
    The expno directory contains all information to describe one experiment. The
    directory name is the experiment number (usually an integer). The data from
    every new mri-sequence executed during the MR scan is stored in a new expno
    directory.
    <procno>
    The procno directory contains all information about a reconstructed or derived
    image series of the experiment. Several procnos may exist which contain
    different derived image series (e.g. different reconstructions, parameter
    images, etc.). The directory name is the processed images number.

    This class reads in a single <expno> directory.

    Attributes
    ----------
    TODO

    self.timestamp : datetime.datetime
        The time point at which the scan...
            ...finished running.
            ...was aborted.
            ...was last edited without being executed.
            ...was created.
        Note: You can actually compare two BrukerExp objects using '<' and '>'
              to find out which one has the latest timepoint.


    METHODS
    -------
    Load_fid-file()

    Load_rawdatajob0_file()

    Load_2dseq_file()
    """

    def __init__(self, source, load_data=True, log_mode="warning"):
        """
        Initializer of the class.

        Parameters
        ----------
        source : str or BrukerExp-object
            The complete path to measurement/experiment folder
            In case 'source' is a BrukerExp object the path is automatically
            extracted form the object. That makes reloading a BrukerExp easier.

        Raises
        ------
        FileNotFoundError
            If path provided does not lead to a directory containing certain
            files identifying it as a bruker scan folder. One of the following
            key files must be included: "pdata", "fid", "acqp" or "seq2d".
        """
        logger.setLevel(LOG_MODES[log_mode])

        exp_folder_path = source.path if isinstance(source, BrukerExp) else source

        if self.__is_valid_scan_dir(exp_folder_path):
            # store path
            self.path = Path(exp_folder_path)
            # load meta data files
            self.method = self.__ReadParamFile("method")
            self.acqp = self.__ReadParamFile("acqp")
            # get VisuExperimentNumber (E-number)
            self.ExpNum = self.__getVisuExperimentNumber()
            # define the measurement type
            self.type = self.__GetMeasurementType()
            # list of names of all recon folders inside 'pdata'
            self.recon_names = self.__FindRecons()
            # add number of receiver channels
            self.n_receivers = self.method["PVM_EncNReceivers"]
            # the time the BrukerExp object was last edited on the scanner
            self.timestamp = self.method["MetaTimeStamp"]

            if load_data:
                # load raw data into the instance
                self.fid = self.Load_fid_file()
                self.rawdatajob0 = self.Load_rawdatajob0_file()
                # load reconstructed data stored in '.../pdata/1'
                self.seq2d = self.Load_2dseq_file()
        elif self.__init_dummy(exp_folder_path):
            # Initialize dummy sequence
            base_dir = Path(__file__).resolve().parents[2]
            # set path to directory where the stock_method_file sits:
            self.path = base_dir / "examples"
            # load meta data files
            self.method = self.__ReadParamFile("dummy_method_file")
            self.acqp = self.__ReadParamFile("dummy_acqp_file")

            # get VisuExperimentNumber (E-number)
            self.ExpNum = -1
            # define the measurement type
            self.type = "dummy sequence"
            # list of names of all recon folders inside 'pdata'
            self.recon_names = []
            # add number of receiver channels
            self.n_receivers = 0
            # the time the BrukerExp object was last edited on the scanner
            self.timestamp = np.nan
            if load_data:
                # load raw data into the instance
                self.fid = np.zeros((128 * 14 * 12 * 2, 1), dtype="complex")
                self.rawdatajob0 = np.zeros((128 * 14 * 12 * 2, 1), dtype="complex")
                # load reconstructed data stored in '.../pdata/1'
                self.seq2d = np.zeros((256, 14, 12, 1))
        else:
            raise FileNotFoundError(
                f"Folder {exp_folder_path} does not exist or is not a bruker experiment."
            )

    def __repr__(self):
        return f"<{__package__}.{self.__class__.__name__} name={self.acqp['ACQ_scan_name']}>"

    def __str__(self):
        return f"{self.acqp['ACQ_scan_name'][1:-1]}"  # drop encasing arrows <>

    def __gt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp > other.timestamp

    def __lt__(self, other):
        """This makes it easy to sort multiple experiments by their timestamp."""
        return self.timestamp < other.timestamp

    @staticmethod
    def __is_valid_scan_dir(path) -> bool:
        """Check if directory provided is a valid bruker scan directory."""
        if not os.path.isdir(path):
            return False

        content = os.listdir(path)

        if len(set(content).intersection(["pdata", "fid", "acqp", "method"])) > 0:
            if "acqp" not in content:
                logger.critical(
                    f"Skipping directory '{path}', which seems to be a scan directory but 'acqp' file is missing."
                )
                return False
            if "method" not in content:
                logger.critical(
                    f"Skipping directory '{path}', which seems to be a scan directory but 'method' file is missing."
                )
                return False
            return True
        return False

    @staticmethod
    def __is_valid_recon_dir(path) -> bool:
        """Check if directory provided is a 'pdata' reconstruction directory."""
        if os.path.isdir(path):
            content = os.listdir(path)
            if len(set(content).intersection(["2dseq"])) > 0:
                return True
        return False

    def __getVisuExperimentNumber(self) -> int:
        """Try to get experiment number. Returns -1 when fails.

        First try to convert folder name into experiment number.

        In case that fails, falls back to extracting the number from the
        ACQ_scan_name (see self.__str__) using regular expressions.

        In case no experiment number is found inside the name, return -1.
        """
        # try to get ExpNum from scan folder name
        if self.path.name.isdigit():
            return int(self.path.name)

        # try to match last (E...) in scan name using regex
        match = re.match(r".*\(E(\d+)\)", str(self))

        return int(match.group(1)) if match else -1

        # # search for visu_pars file, if it exists extract VisuExperimentNumber:
        # # ISSUE: sometimes number is greater than 90000
        # visufname = self.path / "visu_pars"
        # if os.path.isfile(visufname):
        #     with open(visufname, "r") as f:
        #         for line in f:
        #             if line.startswith("##$VisuExperimentNumber"):
        #                 return int(line.split("=")[1].strip())

    @staticmethod
    def __ParseSingleValue(val: str):
        """Extract a single value from a string.

        Args:
            value: A string containing a single value.

        Returns:
            An integer, a float, or a string without a newline character.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.
        """
        # check if int
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

        # check if float
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

        # otherwise remove newline character and return string
        return val.rstrip("\n")

    @staticmethod
    def __ParseArray(current_file, line):
        """Extract an array from a line in a file.

        Args:
            current_file: A file object to read from.
            line: A string containing an array definition.

        Returns:
            A NumPy array or a single value if the array only contains one element.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.
        """
        # extract the arraysize and convert it to numpy
        line = line[1:-2].replace(" ", "").split(",")
        array_size = np.array([int(x) for x in line])

        # then extract the next line
        val_list = current_file.readline().split()

        # if the line was a string, then return it directly
        try:
            float(val_list[0])
        except (TypeError, ValueError):
            return " ".join(val_list)

        # include potentially multiple lines
        while len(val_list) != np.prod(array_size):
            val_list += current_file.readline().split()

        # try converting to int, if error, then to float
        try:
            val_list = [int(x) for x in val_list]
        except ValueError:
            val_list = [float(x) for x in val_list]

        # convert to numpy array
        if len(val_list) > 1:
            return np.reshape(np.array(val_list), array_size)
        # or to plain number
        return val_list[0]

    def __ReadParamFile(self, file_name: str) -> dict:
        """
        Reads a Bruker MRI experiment's method or acqp file to a dictionary.

        Args:
            file_name: A string containing the name of the file to read.

        Returns:
            A dictionary containing the parsed parameters and their values.

        Inspired by BrukerMRI package. https://github.com/jdoepfert/brukerMRI.
        """
        param_dict = {}
        file_path = self.path / file_name

        if not file_path.exists() or os.path.getsize(file_path) == 0:
            logger.error(f"File '{file_path}' does not exist or is empty.")
            return param_dict

        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break

                # when line contains parameter
                if line.startswith("##$"):
                    (param_name, current_line) = line[3:].split("=")  # split at "="

                    # if current entry (current_line) is arraysize
                    if current_line[0:2] == "( " and current_line[-3:-1] == " )":
                        value = self.__ParseArray(f, current_line)

                    # if current entry (current_line) is struct/list
                    elif current_line[0] == "(" and current_line[-3:-1] != " )":
                        # if necessary read in multiple lines
                        while current_line[-2] != ")":
                            current_line = current_line[0:-1] + f.readline()

                        # parse the values to a list
                        value = [
                            self.__ParseSingleValue(x)
                            for x in current_line[1:-2].split(", ")
                        ]
                    # otherwise current entry must be single string or number
                    else:
                        value = self.__ParseSingleValue(current_line)

                    # save parsed value to dict
                    param_dict[param_name] = value

                # This exists only to extract the TimeStamp from the method file
                if line.startswith("$$ File finished by PARX at"):
                    time_str = line.split("at")[1].strip()
                    format_str = "%Y-%m-%d %H:%M:%S.%f %z"

                    time_obj = datetime.strptime(time_str, format_str)

                    param_dict["MetaTimeStamp"] = time_obj

        return param_dict

    def __GetMeasurementType(self) -> str:
        """Returns measurement type from method file as string"""
        return self.method["Method"]

    def __FindRecons(self):
        """Find all recon-folder in pdata and return their names in a list."""
        pdata_path = self.path / "pdata"
        pdata_content = os.listdir(pdata_path)

        recon_folder_names = []

        for recon_name in pdata_content:
            if self.__is_valid_recon_dir(pdata_path / recon_name):
                recon_folder_names.append(recon_name)

        recon_folder_names.sort()

        return recon_folder_names

    def Load_fid_file_update(self):
        raw_file = None

        fid_path = self.path / "fid"
        ser_path = self.path / "ser"

        nr = self.method['PVM_NRepetitions']
        ni = self.acqp["NI"]
        chans = self.bruker_get_selected_receivers()

        matrix = self.acqp['ACQ_size']
        numDataHighDim = np.prod(matrix[1:-1])
        bytorda = self.acqp["BYTORDA"]
        data_format = self.acqp["GO_raw_data_format"]
        def get_endianess(bytorda):
            if bytorda == "little":
                endian = '<'
            elif bytorda == "big":
                endian = '>'
            else:
                endian = '<'
            return endian

        def get_format_and_bits(data_format):
            if data_format == 'GO_32BIT_SGN_INT':
                format = 'int32'
                bits = 32
            elif data_format == 'GO_16BIT_SGN_INT':
                format = 'int16'
                bits = 16
            elif data_format == 'GO_32BIT_FLOAT':
                format = 'float32'
                bits = 32
            else:
                format = 'int32'
                print('Data-Format not correctly specified! Set to int32')
                bits = 32

            return format, bits

        format, bits = get_format_and_bits(data_format)
        endian = get_endianess(bytorda)


        if self.acqp["GO_block_size"] == 'Standard_KBlock_Format':
            blockSize = (np.ceil(matrix[0] * chans * (bits / 8) / 1024) * 1024 / (bits / 8)).astype(int)
        else:
            blockSize = (matrix[0] * chans).astype(int)

        def read_file(file_path, block_size, num_data_high_dim, NI, NR, format_str, endian, single_bool):
            # Define the dtype based on the format and endianness
            if format_str == 'int32':
                dtype = np.dtype(f'{endian}i4')
            elif format_str == 'int16':
                dtype = np.dtype(f'{endian}i2')
            elif format_str == 'float32':
                dtype = np.dtype(f'{endian}f4')
            else:
                raise ValueError(f"Unsupported format: {format_str}")

            # Open the file and read the data
            with open(file_path, 'rb') as file:
                if single_bool:
                    # Read and convert to float32
                    fid_file = np.fromfile(file, dtype=dtype, count=block_size * num_data_high_dim * NI * NR)
                    fid_file = fid_file.astype(np.float32)
                else:
                    # Read without conversion
                    fid_file = np.fromfile(file, dtype=dtype, count=block_size * num_data_high_dim * NI * NR)
                # Reshape the array
                fid_file = fid_file.reshape((block_size, num_data_high_dim * NI * NR), order='F')
            return fid_file

        if fid_path.exists() and os.path.getsize(fid_path) > 0:
            try:
                print(f"file_path={fid_path}, block_size={blockSize}, num_data_high_dim={numDataHighDim}, NI={ni}, "
                      f"NR={nr}, format_str={format}, endian={endian}, single_bool={False}")
                raw_file = read_file(file_path=fid_path, block_size=blockSize, num_data_high_dim=numDataHighDim, NI=ni,
                                     NR=nr, format_str=format, endian=endian, single_bool=False)
            except Exception as e:
                err_msg, err_type = str(e), e.__class__.__name__
                msg = f"The following error occured while loading '{fid_path}':\n"
                msg += f"    {err_type}: {err_msg}"
                logger.error(msg)

    def Load_fid_file(self):
        """Read fid file and return it as np.array.

        If multi slice data is loaded into topspin (i.e. SP with repetitions)
        the fid file is replaced by the ser file, which is loaded similarly.

        Therefore, in case there is no fid file but an ser file, try to load ser
        instead.

        Returns
        -------
        np.array
            complex fid file
        """
        raw_file = None

        fid_path = self.path / "fid"
        ser_path = self.path / "ser"

        if fid_path.exists() and os.path.getsize(fid_path) > 0:
            with open(fid_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                except Exception as e:
                    err_msg, err_type = str(e), e.__class__.__name__
                    msg = f"The following error occured while loading '{fid_path}':\n"
                    msg += f"    {err_type}: {err_msg}"
                    logger.error(msg)

        elif ser_path.exists() and os.path.getsize(ser_path) > 0:
            with open(ser_path, "rb") as f:
                try:
                    raw_file = np.fromfile(f, dtype=np.int32)
                except Exception as e:
                    err_msg, err_type = str(e), e.__class__.__name__
                    msg = f"The following error occured while loading '{ser_path}':\n"
                    msg += f"    {err_type}: {err_msg}"
                    logger.error(msg)
        else:
            logger.error(
                f"Both 'fid' and 'ser' files do not exist or are empty for scan {self.ExpNum}."
            )

        if raw_file is None:
            return np.array([])

        # turn into complex array and return
        return raw_file[0::2] + 1j * raw_file[1::2]

    def bruker_get_selected_receivers(self, *args):
        """ Inspired by the Matlab tool provided by Bruker."""
        if len(args) == 1:
            if 'ACQ_ReceiverSelectPerChan' in self.acqp:
                job_index = args[0]
                job_chan = self.acqp['ACQ_jobs'][7][job_index]  # Python uses 0-based indexing
                num_selected_receivers = 0
                for rec_num in range(self.acqp['ACQ_ReceiverSelectPerChan'].shape[1]):
                    if self.acqp['ACQ_ReceiverSelectPerChan'][job_chan[0], rec_num, :].lower() == 'yes':
                        num_selected_receivers += 1
                return num_selected_receivers
            else:
                raise ValueError(
                    'receive selection per job not supported by dataset. please remove job index from function call.')
        elif len(args) == 0:
            if self.acqp['ACQ_experiment_mode'] == 'ParallelExperiment':
                if 'GO_ReceiverSelect' in self.acqp:
                    if isinstance(self.acqp['GO_ReceiverSelect'], str):
                        return sum(1 for row in self.acqp['GO_ReceiverSelect'] if row.lower() == 'yes')
                    else:
                        return sum(1 for x in self.acqp['GO_ReceiverSelect'] if x.lower() == 'yes')
                elif 'ACQ_ReceiverSelect' in self.acqp:
                    if isinstance(self.acqp['ACQ_ReceiverSelect'], str):
                        return sum(1 for row in self.acqp['ACQ_ReceiverSelect'] if row.lower() == 'yes')
                    else:
                        return sum(1 for x in self.acqp['ACQ_ReceiverSelect'] if x.lower() == 'yes')
                else:
                    print('No information about active channels available.')
                    print('The number of channels is set to 1 ! But at this point the only effect is a bad matrixsize.')
                    print('Later you can change the size yourself.')
                    return 1
            else:
                return 1
        else:
            raise ValueError('too many input arguments!')

    def Load_rawdatajob0_file(self):
        """
        Read the rawdata.job0 file and return it as a np.array.

        Returns
        -------
        complex_file: complex fid
        """
        file_path = self.path / "rawdata.job0"

        if file_path.exists() and os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as f:
                raw_file = np.fromfile(f, dtype=np.int32, count=-1)
                complex_file = raw_file[0::2] + 1j * raw_file[1::2]
        else:
            # only info-level since this is quite often the case.
            logger.info(
                f"'rawdata.job0' file does not exist or is empty for scan {self.ExpNum}."
            )
            complex_file = np.array([])

        return complex_file

    def Load_2dseq_file(self, recon_num=1):
        """
        Load 2dseq file and return it as np.array.

        To load in a specific reconstruction use the argument recon_num.
        Note: This method makes use of the brukerapi package.

        Parameters
        ----------
        recon_num : int, optional
            Name/number of reconstruction, by default 1.

        Returns
        -------
        out : np.array
            Complex array of the 2dseq file.
        """
        file_path = self.path / "pdata" / str(recon_num) / "2dseq"
        if file_path.exists() and os.path.getsize(file_path) > 0:
            return Dataset(file_path).data
        else:
            # only info-level since this happens quite often and recon numbers
            # are already displayed with brukerdir by default.
            logger.info(
                f"'2dseq' file does not exist or is empty for scan {self.ExpNum}."
            )
            return np.array([])

    def open(self, paramfile="method"):
        """Open method file in default text editor for '.json' files."""
        import sys, time, tempfile, pprint, subprocess

        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)

        def open_file_platfrom_independent(filename):
            if sys.platform == "win32":
                os.startfile(filename)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, filename])

        try:
            # print(f"Dumping temporary file here:", tmp.name)
            pretty_str = pprint.pformat(getattr(self, paramfile))
            tmp.write(pretty_str.encode("ascii"))
        finally:
            tmp.close()
            open_file_platfrom_independent(tmp.name)
            time.sleep(2)
            os.unlink(tmp.name)

    def paper_summary(self):
        def extract_parameter_list(paramlist, title):
            data = []  # List to hold data for DataFrame creation

            def get_from_method(item):
                key, edit_func = item
                return edit_func(self.method[key])

            # Iterate through paramlists to collect data and populate dataframe
            for description, access_item in paramlist.items():
                try:
                    value = str(get_from_method(access_item))
                    data.append(value)
                except KeyError:
                    data.append("-")

            # Create MultiIndex DataFrame using collected data
            index = pd.MultiIndex.from_tuples(
                [(title, d) for d in paramlist.keys()],
                names=["Category", "Description"],
            )
            df = pd.DataFrame(data, index=index, columns=["Value"])

            return df

        # Define your parameter lists

        paramlist_Hz = {
            "Reference Frequency": ("PVM_FrqRef", lambda x: x[0]),
            "Working Frequency": ("PVM_FrqWork", lambda x: x[0]),
            "Working Frequency Offset": (
                "PVM_FrqWorkOffset",
                lambda x: f"{x[0]:g}E-06",
            ),
        }

        paramlist_ppm = {
            "Reference Frequency": ("PVM_FrqRefPpm", lambda x: f"{x[0]:g}"),
            "Working Frequency": ("PVM_FrqWorkPpm", lambda x: f"{x[0]:g}"),
            "Working Frequency Offset": ("PVM_FrqWorkOffsetPpm", lambda x: f"{x[0]:g}"),
        }

        paramlist_timings = {
            "Echo Acquisition Mode": ("EchoAcqMode", lambda x: x),
            "Number of Echo Images": ("PVM_NEchoImages", lambda x: x),
            "Large Delta TE": ("EchoSpacing", lambda x: x),
            "Small Delta TE": ("SmallDeltaTE", lambda x: x),
            "Small Delta TE Position": ("SmallDeltaTEPosition", lambda x: x),
            "Echo Time [ms]": ("PVM_EchoTime", lambda x: x),
            "Repetition Time": ("PVM_RepetitionTime", lambda x: x),
            "Total Scan Time": ("PVM_ScanTimeStr", lambda x: x),
            "Repetitions": ("PVM_NRepetitions", lambda x: x),
            "Averages": ("PVM_NAverages", lambda x: x),
        }

        paramlist_acquisition = {
            "Acquisition FOV [mm]": ("PVM_Fov", lambda x: x),
            "Acquisition Matrix": ("PVM_Matrix", lambda x: x),
            "Slice Thickness [mm]": ("PVM_SliceThick", lambda x: x),
            "Exc. Pulse: Flip Angle": ("ExcPulse1", lambda x: x[2]),
            "Exc. Pulse: BW": ("ExcPulse1", lambda x: x[1]),
            "Exc. Pulse: Shape": (
                "ExcPulse1",
                lambda x: x[11].replace(".exc>", "").replace("<", ""),
            ),
            "Acq. BW": ("PVM_EffSWh", lambda x: x),
            "Acq. Encoding Order": ("PVM_EncOrder", lambda x: x.replace("_ENC", "")),
        }

        paramlist_saturation = {
            "FOV Sat Module: OnOff": ("PVM_FovSatOnOff", lambda x: x),
            "FOV Sat Module: Slices": ("PVM_FovSatNSlices", lambda x: x),
            "FOV Sat Spoiler: Amplitude": ("PVM_FovSatSpoil", lambda x: x[2]),
            "FOV Sat Spoiler: Duration": ("PVM_FovSatSpoil", lambda x: x[3]),
            "FOV Sat Pulse: BW": ("PVM_FovSatPul", lambda x: x[1]),
            "FOV Sat Pulse: Shape": (
                "PVM_FovSatPulEnum",
                lambda x: x.replace(">", "").replace("<", ""),
            ),
            "WaterSuppr Module: Mode": ("PVM_WsMode", lambda x: x),
            "WaterSuppr Module: BW": ("PVM_WsBandwidth", lambda x: x),
            "WaterSuppr Spoiler: Amplitude": (
                "PVM_ChSpoilerStrength",
                lambda x: np.round(x),
            ),
            "WaterSuppr Spoiler: Duration": ("PVM_ChSpoilerOnDuration", lambda x: x),
            "WaterSuppr Pulse: BW": ("PVM_ChPul1", lambda x: x[1]),
            "WaterSuppr Pulse: Shape": (
                "PVM_ChPul1",
                lambda x: x[11].replace(".exc>", "").replace("<", ""),
            ),
            "FatSat OnOff": ("PVM_FatSupOnOff", lambda x: x),
        }

        paramlist_reco = {
            "Channel Combination": ("RecoCombineMode", lambda x: x),
        }
        paramlist_spectroscopy = {
            "Spectral Points": ("PVM_SpecMatrix", lambda x: x),
            "Acq. BW": ("PVM_SpecSWH", lambda x: x),
            "Slice Selection": ("SliceSelOnOff", lambda x: x),
            "Slice Thickness": ("SliceThick", lambda x: x),
        }

        paramlist_press = {
            "Voxel Size": ("PVM_VoxArrSize", lambda x: x[0]),
            "PVM_EchoTime": ("PVM_EchoTime", lambda x: x),
        }

        parameter_lists = {
            "Freq. Hz": paramlist_Hz,
            "Freq. PPM": paramlist_ppm,
            "Sequence Timings": paramlist_timings,
            "Acquisition": paramlist_acquisition,
            "Saturation": paramlist_saturation,
            "Spectroscopy": paramlist_spectroscopy,
            "PRESS": paramlist_press,
            "RECO": paramlist_reco,
        }

        frame_collection = []
        for name, plist in parameter_lists.items():
            df = extract_parameter_list(plist, name)
            frame_collection.append(df)

        df_ALL = pd.concat(frame_collection)

        styles = [
            {"selector": "th", "props": [("font-size", "109%")]},  # tabular headers
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        df_styler = df_ALL.style.set_table_styles(styles)

        display(HTML(df_styler.to_html()))

    def __init_dummy(self, path):
        """Check if a dummy function should be initialized"""
        if path == "dummy":
            return True
        else:
            return False
