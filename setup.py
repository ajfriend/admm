from setuptools import setup, find_packages

setup(
    name='admm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Framework for ADMM',
    long_description="Framework for ADMM",

    # The project's main homepage.
    url='https://github.com/pypa/sampleproject',

    # Author details
    author='AJ Friend',
    author_email='ajfriend@gmail.com',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='admm convex optimization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'cvxpy', 'matplotlib'],


)