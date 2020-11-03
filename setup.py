from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='im3vis',
    version='0.1.0',
    packages=['im3vis'],
    url='https://github.com/IMMM-SFA/im3vis',
    license='BSD 2-Clause',
    author='Chris R. Vernon',
    author_email='chris.vernon@pnnl.gov',
    description='Jupyter notebooks to generate common visualization tasks',
    long_description=readme(),
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=get_requirements()
)