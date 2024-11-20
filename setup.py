from setuptools import find_packages, setup

setup(
    name="AMP",                  # Name of your package
    version="0.0.2",                           # Package version
    description="ascend model patcher",
    long_description=open("README.md").read(),  # Long description from README
    long_description_content_type="text/markdown",  # Description format
    author="guozr",                        # Author's name
    author_email="guozr1997@hotmail.com",     # Author's email
    url="https://github.com/starmountain1997/AMP",
    # URL for your project (e.g., GitHub)
    packages=find_packages(include=["amp", "amp.*"]),
    install_requires=[                         # Dependencies
    ],
    classifiers=[                              # Classifiers for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',                   # Minimum Python version
)
