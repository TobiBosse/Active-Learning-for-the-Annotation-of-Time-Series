from setuptools import setup

setup(name='salat', # Super Active Learning At Timeseries
      version= '0.5',
      author = 'Tobias Bosse',
      install_requires=[
          'numpy',
          'scipy', 
          'matplotlib',
          'sklearn', 
          'modAL', # AL env
          'pyts' #contains BOSS implementation
          ]  # And any other dependencies foo needs
)
# SALAT is based on BOSS and ROCKET