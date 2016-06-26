try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='convcuda',
    description='Convolutional Networks implemented in CUDA.',
    long_description=open('README.md').read(),
    version='0.1',
    packages=['convcuda'],
    scripts=[],
    author='Lucas David, Paulo Finaridi',
    author_email='lucasolivdavid@gmail.com',

    url='https://github.com/lucasdavid/convolutional-cuda',
    download_url='https://github.com/lucasdavid/convolutional-cuda/archive/master.zip',
    install_requires=['numpy', 'pycuda', 'scikit-learn'],
    tests_require=open('requirements-dev.txt').readlines(),
)
