import unittest
import numpy as np
import pandas as pd
import numpy.testing as np_testing
import pandas.testing as pd_testing
import io, os, sys, types
from IPython import get_ipython
from nbformat import current
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow import random


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
        import Activity3_01
        self.activity = Activity3_01
        
        dirname = self._dirname_if_file('../data/outlier_feats.csv')
        self.feats_loc = os.path.join(dirname, 'outlier_feats.csv')
        self.target_loc = os.path.join(dirname, 'outlier_target.csv')
        
        self.feats = pd.read_csv(self.feats_loc)
        self.target = pd.read_csv(self.target_loc)
        
        seed = 1
        np.random.seed(seed)
        random.set_seed(seed)
        model_1 = Sequential()
        model_1.add(Dense(1, activation='sigmoid', input_dim=2)) 
        model_1.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_1.fit(self.feats, self.target, batch_size=5, epochs=100, verbose=0, validation_split=0.2, shuffle=False)
        self.test_loss_1 = model_1.evaluate(self.feats, self.target)
        self.activity_test_loss_1 = self.activity.model_1.evaluate(self.activity.feats, self.activity.target)
        
        np.random.seed(seed)
        random.set_seed(seed)
        model_2 = Sequential() 
        model_2.add(Dense(3, activation='relu', input_dim=2))
        model_2.add(Dense(1, activation='sigmoid'))
        model_2.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_2.fit(self.feats, self.target, batch_size=5, epochs=200, verbose=0, validation_split=0.2, shuffle=False)
        self.test_loss_2 = model_2.evaluate(self.feats, self.target)
        self.activity_test_loss_2 = self.activity.model_2.evaluate(self.activity.feats, self.activity.target)
        
        np.random.seed(seed)
        random.set_seed(seed)
        model_3 = Sequential() 
        model_3.add(Dense(6, activation='relu', input_dim=2))
        model_3.add(Dense(1, activation='sigmoid'))
        model_3.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_3.fit(self.feats, self.target, batch_size=5, epochs=400, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_3 = model_3.evaluate(self.feats, self.target)
        self.activity_test_loss_3 = self.activity.model_3.evaluate(self.activity.feats, self.activity.target)
        
        np.random.seed(seed)
        random.set_seed(seed)
        model_4 = Sequential() 
        model_4.add(Dense(3, activation='tanh', input_dim=2))
        model_4.add(Dense(1, activation='sigmoid'))
        model_4.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_4.fit(self.feats, self.target, batch_size=5, epochs=200, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_4 = model_4.evaluate(self.feats, self.target)
        self.activity_test_loss_4 = self.activity.model_4.evaluate(self.activity.feats, self.activity.target)
        
        np.random.seed(seed)
        random.set_seed(seed)
        model_5 = Sequential() 
        model_5.add(Dense(6, activation='tanh', input_dim=2))
        model_5.add(Dense(1, activation='sigmoid'))
        model_5.compile(optimizer='sgd', loss='binary_crossentropy') 
        model_5.fit(self.feats, self.target, batch_size=5, epochs=400, verbose=0, validation_split=0.2, shuffle=False) 
        self.test_loss_5 = model_5.evaluate(self.feats, self.target)
        self.activity_test_loss_5 = self.activity.model_4.evaluate(self.activity.feats, self.activity.target)
        
        
    def test_values(self):
        pd_testing.assert_frame_equal(self.activity.feats, self.feats)
        pd_testing.assert_frame_equal(self.activity.target, self.target)
        np_testing.assert_almost_equal(self.test_loss_1, self.activity_test_loss_1, decimal=1)
        np_testing.assert_almost_equal(self.test_loss_2, self.activity_test_loss_2, decimal=1)
        np_testing.assert_almost_equal(self.test_loss_3, self.activity_test_loss_3, decimal=1)
        np_testing.assert_almost_equal(self.test_loss_4, self.activity_test_loss_4, decimal=1)
        np_testing.assert_almost_equal(self.test_loss_5, self.activity_test_loss_5, decimal=1)

if __name__ == '__main__':
    unittest.main()
