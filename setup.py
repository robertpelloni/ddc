from setuptools import setup, find_packages
import os

def read_version():
    with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
        return f.read().strip()

setup(
    name='ddc',
    version=read_version(),
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
    include_package_data=True,
    author='Jules (Modernized Port)',
    author_email='jules@example.com',
    url='https://github.com/chrisdonahue/ddc',
)
