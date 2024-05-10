from setuptools import setup, find_packages

setup(
    name='renesasadc',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'librosa',
        'pandas',
        'scikit-learn',
        'seaborn'
    ],
    author='Tanay Nayak',
    author_email='tanay.nayak@gmail.com',
    description='An audio processing package for feature extraction and classification.',
    python_requires='>=3.8',
)
