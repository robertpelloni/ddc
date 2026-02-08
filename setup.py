from setuptools import setup, find_packages

<<<<<<< HEAD
setup(
    name='ddc',
    version='0.1.0',
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
    author='Jules (Modernized Port)',
    author_email='jules@example.com',
    url='https://github.com/chrisdonahue/ddc',
)
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
