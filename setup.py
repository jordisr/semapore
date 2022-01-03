import setuptools

setuptools.setup(
   name='semapore',
   version='0.0.0',
   author='Jordi Silvestre-Ryan',
   packages=['semapore', 'semapore.util'],
   entry_points={'console_scripts':['semapore = semapore.__main__:main']}
)
