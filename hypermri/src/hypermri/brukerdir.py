import warnings

from . import sequences
from .brukerexp import BrukerExp
from .utils.utils_logging import LOG_MODES, init_default_logger

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from collections.abc import Mapping
from IPython.display import display
import matplotlib.colors as mcolors

logger = init_default_logger(__name__)


class BrukerDirError(Exception):
    pass


class BrukerDir(Mapping):
    """Dictionary-like class to handle scan collections from given directory.

    Parameters
    ----------
    path : string
        Path to the folder containing all scans of the study to be loaded in.

    keywords : string or tuple or list, optional
        Add additional columns to the internal dataframe. See
        BrukerDir.add_column and BrukerDir.add_columns for more information.
         - 'keyword' is string or tuple --> BrukerDir.add_column
         - 'keyword' is list --> BrukerDir.add_columns
        Default is None.

    sort : bool, optional
        BrukerExp objects are sorted by their time-stamp by default. In case of
        sort=False, they are sorted by their 'E-number' (experiment number).

    verbose : bool, optional
        Wether or not BrukerDir.display is called to generate a nice scan over-
        view. Default is True.

    smart_load : bool, optional
        When set to true scans are not read in as BrukerExp classes but as
        custom sequence classes whereever possible. The configuration which scan
        is loaded in as which class is specified in the config file:
        'config/SmartLoadMeasurements.json'. Default is True.

    load_data : bool, optional
        Default is True. Can be set to False only when smart_load is turned off.
        In this case, all scans are loaded in as BrukerExp objects with only
        the parameter files loaded in.

    log_mode : string
        See utils.logging for more information. Default is set to 'error'.
    """

    def __init__(
        self,
        path,
        keywords=None,
        verbose=True,
        smart_load=True,
        load_data=True,
        log_mode="error",
        colorize=True,
        sort=True,
    ):
        logger.setLevel(LOG_MODES[log_mode])

        self.path = Path(path)

        if smart_load:
            assert load_data == True, "No control on data loading during smart load!"
            # preliminary read in of scans, we want the param files only
            self.__scans = self.__ReadInMeasurements(load_data=False)
            # do a smart reload of the scans
            self.__SmartLoadMeasurements()
        else:
            # read in all scans as BrukerExp objects
            self.__scans = self.__ReadInMeasurements(load_data)

        self.new_column_from_scanobj(
            "Reconstructions", lambda x: x.recon_names if x.recon_names else "MISSING"
        )

        self.date = self.__GetSubjectDate()
        self.subject = {}
        self.subject["subject_name"] = self.__GetSubjectName()
        self.subject["subject_dob"] = self.__GetSubjectDOB()
        self.subject["subject_weight"] = self.__GetSubjectWeight()

        if isinstance(keywords, (str, tuple)):
            self.add_column(keywords)

        if isinstance(keywords, list):
            self.add_columns(keywords)

        if sort:
            # the __lt__ and __gt__ attributes of the BrukerExp allow for this
            # nice one-liner to sort the rows by the timestamp.
            self.__scans.sort_values(axis=0, by="ScanObject", inplace=True)

        if verbose:
            self.display(colorize=colorize)

    def __getitem__(self, key):
        if key in self.__scans.index:
            return self.__scans.loc[(key, "ScanObject")]
        else:
            raise BrukerDirError(f"{key} is not a valid key.")

    def __iter__(self):
        return iter(self.__scans["ScanObject"])

    def __len__(self):
        return len(self.__scans)

    def __str__(self):
        return str(self.__scans)

    @property
    def dataframe(self):
        return self.__scans

    @dataframe.setter
    def dataframe(self, df):
        if isinstance(df, pd.DataFrame):
            self.__scans = df
        else:
            raise BrukerDirError(
                "BrukerDir.dataframe must be a pandas DataFrame object."
            )

    def colorize_col(self, col, df_styler=None, opacity=0.5):
        assert opacity <= 1 and opacity >= 0, "Opacity must be in range [0, 1]"

        if df_styler is None:
            df_styler = self.__scans.style

        # if the column item is not hashable, abort
        try:
            values = set(self.__scans[col].values)
        except:
            return

        # convert opacity from rgb or [0,1] to hex
        opacity = mcolors.to_hex((opacity,) * 3)[-2:]

        # if there are more values than colors, abort
        if len(values) <= len(mcolors.TABLEAU_COLORS):
            pltcolors = mcolors.TABLEAU_COLORS
        elif len(values) <= len(mcolors.CSS4_COLORS):
            pltcolors = mcolors.CSS4_COLORS
        else:
            return

        colors = [c + opacity for c in pltcolors.values()]

        # generate color look up table
        color_lut = {k: v for k, v in zip(values, random.sample(colors, len(values)))}
        color_lut["MISSING"] = ""

        f = lambda v: f"background-color: {color_lut[v]};"

        return df_styler.applymap(f, subset=[col])

    def remove(self, scan=None):
        """
        Removes a scan with number scan_num from the object

        Parameters
        ----------
        scan: int
            Experiment number of the scan to be removed

        Returns
        -------
        None
        """
        if scan is None:
            return None

        assert isinstance(scan, int), "scan_num has to be an integer"

        # check if desired scan for removal is in brukerdir
        if scan in self.__scans.index:
            # remove scan from scans:
            self.__scans.drop(index=scan, inplace=True)
        else:
            warnings.warn(f"scan {scan} does not exist in scans.")

    def display(self, colorize=True):
        """Display all scan-objects using pandas style functionalities."""
        styles = [
            {"selector": "th", "props": [("font-size", "109%")]},  # tabular headers
            {"selector": "td", "props": [("font-size", "107%")]},  # tabular data
        ]

        df_styler = self.__scans.style.set_table_styles(styles)

        if self.date:
            date_string = self.date.strftime("%A %d. %B %Y  %H:%M:%S")
            df_styler = df_styler.set_caption(date_string)

        if colorize:
            # skip first three columns, try to colorize all others
            for col in self.__scans.columns[3:]:
                self.colorize_col(col, df_styler, opacity=0.3)

        def color_MISSING(cell_value):
            # cell_value == "MISSING" raises a warning if cell_value is an array
            if isinstance(cell_value, str) and cell_value == "MISSING":
                return "color: red"
            return ""

        df_styler = df_styler.applymap(color_MISSING)

        display(df_styler)

    def get_available_keys(self):
        keys = []
        for key in self.__scans.index:
            keys.append(key)
        return keys

    def new_column_from_scanobj(self, colname, func):
        self.__scans[colname] = self.__scans["ScanObject"].apply(func)

    def add_column(self, keyword, paramfile="method"):
        """Adds column with value of 'keyword' from 'paramfile' of a scan-object.

        This method tries to extract a single key ('keyword') from the specified
        parameter file of each ScanObject. In case the keyword is not found, the
        entry is set to 'MISSING'.

        Parameter
        ----------
        keyword: string or tuple
            The key of a key-value pair of the specified 'paramfile'.
            - 'keyword' is a string:
                scanobject.paramfile[keyword]
                is retrieved for every ScanObject and saved into a new column.

            - 'keyword' is a tuple:
                kw, idx = keyword
                scanobject.paramfile[kw][idx]
                is retrieved for every ScanObject and saved into a new column.
                This is especially usefull in case the keyword entry of the
                parmfile contains a large array where only certain entries are
                of interest.

        paramfile: string, optional
            Name of the ScanObjects parameter file in which 'keyword' is stored
            as a key-value pair. Default is the method file.
        """

        def get_value(scan_obj):
            """Helper function to get the value of the keyword from the paramfile"""
            if not hasattr(scan_obj, paramfile):
                return "MISSING"
            lookup = getattr(scan_obj, paramfile)
            if isinstance(keyword, tuple):
                kw, idx = keyword
                return lookup[kw][idx] if kw in lookup else "MISSING"
            else:
                return lookup[keyword] if keyword in lookup else "MISSING"

        self.new_column_from_scanobj(colname=keyword, func=get_value)

    def add_columns(self, keyword_list, paramfile="method"):
        """A version of BrukerDir.add_column that takes in a list of keywords."""
        if not isinstance(keyword_list, list):
            raise BrukerDirError("Argument must be of type 'list'.")

        for keyword in keyword_list:
            self.add_column(keyword, paramfile)

    def __GetSubjectName(self):
        """Retrieve the SubjectName of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_name_string"):
                        # Read the next line
                        subject_name = next(f).strip()[1:-1]
                        return subject_name
        return None

    def __GetSubjectDOB(self):
        """Retrieve the Subject's weight of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_dbirth"):
                        # Read the next line
                        subject_dob = next(f).strip()[1:-1]
                        return subject_dob
        return None

    def __GetSubjectWeight(self):
        """Retrieve the Subject's date of birth of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_weight"):
                        # Read the next line
                        subject_weight = line.split("=")[1].strip()
                        return subject_weight
        return None

    def __GetSubjectDate(self):
        """Retrieve the SubjectDate of our bruker directory."""
        subject_fname = self.path / "subject"
        if subject_fname.is_file():
            with open(subject_fname, "r") as f:
                for line in f:
                    if line.startswith("##$SUBJECT_date"):
                        date = line.split("=")[1].strip()
            return datetime.strptime(date, "<%Y-%m-%dT%H:%M:%S,%f%z>")
        return None

    def __ReadInMeasurements(self, load_data=True) -> pd.DataFrame:
        """Collect all scan folders and read each scan in as BrukerExp object.

        Parameters
        ----------
        load_data: bool
            When set to False only loads the parameter files, not the data files.
            Default is True, which loads in everything.

        Returns
        -------
        out: pandas.DataFrame
            Returns a DataFrame containing all scans. The scans are saved in the
            row 'ScanObject' with their corresponding E-Number as index.
        """
        # import scan-dir-validation method from BrukerExp class
        is_valid_scan_dir = BrukerExp._BrukerExp__is_valid_scan_dir

        # find all folders in given directory that are a valid scan dir
        scan_dir_paths = [p for p in self.path.iterdir() if is_valid_scan_dir(p)]

        # read in all found scan-folders as BrukerExp objects
        all_scans = [BrukerExp(p, load_data=load_data) for p in scan_dir_paths]

        # use the scan experiment number as index/key for our scans. In case
        # none was found, we assign a unique negative number.
        # Additionally, we check wether the key is already assigned (e.g. when
        # the scan folder has been duplicated). In that case we assign a unique
        # negative number, too.
        keys = []
        cntr = -1
        for scan in all_scans:
            if scan.ExpNum > 0 and scan.ExpNum not in keys:
                keys.append(scan.ExpNum)
            else:
                if scan.ExpNum in keys:
                    msg = f"Scan '{scan.path}' has same key as scan {scan.ExpNum}."
                    msg += f"\n\t-> Assign {cntr} as key instead."
                    logger.error(msg)
                else:
                    msg = f"Scan '{scan.path}' has no valid key."
                    msg += f" -> Assign {cntr} as key instead."
                    logger.warning(msg)
                keys.append(cntr)
                cntr -= 1

        # create a DataFrame with 'keys' as index and scan objects as first entry
        # Note: one could add name="KEYS" to pd.Index
        df = pd.DataFrame(all_scans, index=pd.Index(keys), columns=["ScanObject"])

        df = df.sort_index()

        return df

    def __get_config_content(self):
        """Reads in config file for SmartLoaMeasurement method as json object."""
        # read in the configuration file for our smart load
        module_path = Path(__file__).parents[0]
        config_path = module_path / "config/SmartLoadMeasurements.json"

        with open(config_path, "r") as openfile:
            # Reading from json file
            out = json.load(openfile)

        return out

    def __SmartLoadMeasurements(self):
        """Collect all scans and read them in as custom sequence objects.

        Instead of loading in each scan as a BrukerExp object (like
        self.__ReadInMeasurements() does), this method detects whether a scan
        has a corresponding sequence-class and if so loads it in as such. The
        appropriate sequence-class is specified in the configuration file:

           '/hypermri/config/SmartLoadMeasurements.json'

        In case no matching sequence-class is found the method falls back to
        reading the scan in as a BrukerExp object.

        How it works:
            1. Load in each scans metadata only. This is done by using the
               self.__ReadInMeasurements() method with 'load_data=False'.
            2. Use the 'SmartLoadMeasurements.json' config file and try to match
               a corresponding sequence class to each BrukerExp.type
            3. Load in each scan (with data). Either as a custom sequence object
               - if one was specified in the config file - or as BrukerExp class.

        Returns
        -------
        out: pandas.DataFrame
            Returns a DataFrame containing all scans. The scans are saved in the
            row 'ScanObject' with their corresponding E-Number as index.
            Additionally, a row 'SequenceClass' is added. Here, the name of the
            class used to load in the scan is listed.
        """

        def lookup_sequence_class_name(scan) -> str:
            """Returns name of sequence class for given scan obj using the config file."""
            try:
                # In a dict of lists, finds the key corresponding to the list
                # containing x. Returns the first key found.
                return next(
                    k
                    for k, v in sequence_lookup_dict.items()
                    if scan.type == v or scan.type in v
                )
            except StopIteration:
                return "BrukerExp"

        def string2class(classname) -> type:
            """Clean way to convert a string of a class-name into a usable class."""
            return getattr(sys.modules[__package__ + ".sequences"], classname)

        # read in the configuration file for our smart load (as a dictionary)
        sequence_lookup_dict = self.__get_config_content()

        # Use the lookup_sequence_class_name function  to create a column
        # containing the assigned sequence class names for each scan object.
        self.new_column_from_scanobj("SequenceClass", lookup_sequence_class_name)

        # Go through every row of the DataFrame, and reload the 'hollow' scan
        # object using the sequence class specifed in the "SequenceClass" column.
        for i in tqdm(self.__scans.index, leave=False):
            scan = self.__scans.loc[(i, "ScanObject")]
            new_class_name = self.__scans.loc[(i, "SequenceClass")]

            if new_class_name == "BrukerExp":
                new_class = BrukerExp
            else:
                new_class = string2class(new_class_name)

            try:
                start = time.perf_counter()
                self.__scans.loc[(i, "ScanObject")] = new_class(scan.path)
                stop = time.perf_counter()
                logger.debug(f"Scan {i} took {stop-start:.2f}s to load.")
            except Exception as e:
                err_msg, err_type = str(e), e.__class__.__name__

                msg = "The following error occured when trying to load in scan "
                msg += f"{i} as '{new_class_name}':\n    {err_type}: {err_msg}\n"

                logger.error(msg)
