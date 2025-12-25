from setuptools import setup, find_packages

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
