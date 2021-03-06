import setuptools

setuptools.setup(
   name='semapore',
   version='0.0.0',
   author='Jordi Silvestre-Ryan',
   packages=['semapore', 'semapore.util'],
   entry_points={'console_scripts':['semapore = semapore.__main__:main']},
   install_requires=[
   'tensorflow>=2',
   'tensorflow_probability',
   'numpy',
   'scipy',
   'pandas',
   'matplotlib',
   'biopython',
   'pysam',
   'mappy',
   'h5py',
   'tqdm'
   ]
)
