import unittest
import pandas as pd
import pandas.testing as pd_testing
import numpy.testing as np_testing
import io, os, sys, types
from IPython import get_ipython
from nbformat import current
from IPython.core.interactiveshell import InteractiveShell


def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit('.', 1)[-1]
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path


class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, path=None):
        self.shell = InteractiveShell.instance()
        self.path = path

    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)

        print ("importing Jupyter notebook from %s" % path)

        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = current.read(f, 'json')


        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
            for cell in nb.worksheets[0].cells:
                if cell.cell_type == 'code' and cell.language == 'python':
                    # transform the input to executable Python
                    code = self.shell.input_transformer_manager.transform_cell(cell.input)
                    # run the code in themodule
                    exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod


class NotebookFinder(object):
    """Module finder that locates Jupyter Notebooks"""
    def __init__(self):
        self.loaders = {}

    def find_module(self, fullname, path=None):
        nb_path = find_notebook(fullname, path)
        if not nb_path:
            return

        key = path
        if path:
            # lists aren't hashable
            key = os.path.sep.join(path)

        if key not in self.loaders:
            self.loaders[key] = NotebookLoader(path)
        return self.loaders[key]

sys.meta_path.append(NotebookFinder())


class Test(unittest.TestCase):
     
    def _dirname_if_file(self, filename):
        if os.path.isdir(filename):
            return filename
        else:
            return os.path.dirname(os.path.abspath(filename))

    def setUp(self):
        import Exercise1_02
        #from ipynb.fs.full.Exercise1_02 import *
        self.exercise = Exercise1_02
        
        dirname = self._dirname_if_file('../data/OSI_feats.csv')
        self.feats_loc = os.path.join(dirname, 'OSI_feats.csv')
        self.target_loc = os.path.join(dirname, 'OSI_target.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)

        self.feats['is_weekend'] = self.feats['Weekend'].apply(lambda row: 1 if row == True else 0)

        visitor_type_dummies = pd.get_dummies(self.feats['VisitorType'], prefix='VisitorType')
        visitor_type_dummies.drop('VisitorType_Other', axis=1, inplace=True)
        self.feats = pd.concat([self.feats, visitor_type_dummies], axis=1)
        
        month_dummies = pd.get_dummies(self.feats['Month'], prefix='Month')
        month_dummies.drop('Month_Feb', axis=1, inplace=True)
        self.feats = pd.concat([self.feats, month_dummies], axis=1)
        self.feats.drop(['Weekend', 'VisitorType', 'Month'], axis=1, inplace=True, errors='ignore')                 
                                          
        self.target['Revenue'] = self.target['Revenue'].apply(lambda row: 1 if row == True else 0)
    
    def test_feats_df(self):
        pd_testing.assert_frame_equal(self.exercise.data, self.feats)

    def test_target_df(self):
        pd_testing.assert_frame_equal(self.exercise.target, self.target)

if __name__ == '__main__':
    unittest.main()