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
from sklearn.preprocessing import StandardScaler


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
        import Activity3_02
        self.activity = Activity3_02
        
        dirname = self._dirname_if_file('../data/HCV_feats.csv')
        self.X_loc = os.path.join(dirname, 'HCV_feats.csv')
        self.y_loc = os.path.join(dirname, 'HCV_target.csv')
        
        self.X = pd.read_csv(self.X_loc)
        self.y = pd.read_csv(self.y_loc)

        seed=1
        np.random.seed(seed)
        random.set_seed(seed)
        sc = StandardScaler()
        self.X = pd.DataFrame(sc.fit_transform(self.X), columns=self.X.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed)
        
        np.random.seed(seed)
        random.set_seed(seed)
        classifier = Sequential()
        classifier.add(Dense(units = 4, activation = 'tanh', input_dim = self.X_train.shape[1]))
        classifier.add(Dense(units = 2, activation = 'tanh'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(self.X_train, self.y_train, batch_size = 20, epochs = 100, validation_split=0.1, shuffle=False)
        self.test_loss, self.test_acc = classifier.evaluate(self.X_test, self.y_test['AdvancedFibrosis'])
        
    def test_input_frames(self):
        pd_testing.assert_frame_equal(self.activity.X, self.X)
        pd_testing.assert_frame_equal(self.activity.y, self.y)
        
    def test_model_perf(self):
        np_testing.assert_almost_equal(self.activity.test_loss, self.test_loss, decimal=2)
        np_testing.assert_almost_equal(self.activity.test_acc, self.test_acc, decimal=1)

if __name__ == '__main__':
    unittest.main()
