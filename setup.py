from setuptools import setup, find_packages
<<<<<<< HEAD

<<<<<<< HEAD
setup(
    name='ddc',
    version='0.1.0',
=======
import os

def read_version():
    with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
        return f.read().strip()

setup(
    name='ddc',
    version=read_version(),
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
    description='Dance Dance Convolution: Automatic Stepchart Generation',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0',
        'numpy',
        'tqdm',
        'scipy',
        'librosa',
        'mutagen',
        'Pillow',
        'requests',
        'pandas',
        'scikit-learn',
        'simfile',
        'python-dotenv',
        'torch',
        'resampy'
    ],
    scripts=['autochart.py'],
<<<<<<< HEAD
=======
    include_package_data=True,
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
    author='Jules (Modernized Port)',
    author_email='jules@example.com',
    url='https://github.com/chrisdonahue/ddc',
)
<<<<<<< HEAD
=======
setup(name='ddc',
      version='0.1',
      description='Dance Dance Convolution; learn to choreograph music for rhythm games',
      url='https://github.com/chrisdonahue/ddc',
      author='Chris Donahue',
      author_email='cdonahue@ucsd.edu',
      license='MIT',
      packages=['ddc', 'ddc.datasets.sm', 'ddc.models'])
>>>>>>> origin/master_v2
=======
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
