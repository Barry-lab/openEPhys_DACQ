# Python package for neural data acquisition with automated experiments
# Copyright (c) 2017-2020 Barry-lab https://www.ucl.ac.uk/biosciences/people/caswell-barry
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""A package for neural data acquisition with automated experiments"""

import os
import sys
from setuptools import setup, find_packages

if sys.version_info[0] != 3:
    raise ValueError('This package requires Python 3')

HERE = os.path.abspath(os.path.dirname(__file__))

# Workaround <http://www.eby-sarna.com/pipermail/peak/2010-May/003357.html>
try:
    import multiprocessing
except ImportError:
    pass

__project__      = 'openEPhys_DACQ'
__version__      = '0.86'
__author__       = 'Barry-lab'
__author_email__ = 'caswellbarry@ucl.ac.uk'
__url__          = 'https://barry-lab.github.io/openEPhys_DACQ/'

__classifiers__ = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]

__keywords__ = [
    'open-ephys',
    'raspberrypi',
]

__requires__ = [
    'h5py',
    'matplotlib',
    'numpy',
    'opencv-python',
    'paramiko',
    'psutil',
    'pygame',
    'PyQt5',
    'pyqtgraph',
    'pyzmq',
    'scikit-learn',
    'scipy',
    'tqdm',
    'Pillow'
]

__extra_requires__ = {
    'doc':   ['sphinx' , 'sphinx_rtd_theme'],
    'Processing':  ['klustakwik', 'kilosort', 'matlab.engine']
}

__entry_points__ = {
    'console_scripts': [
        'openEPhys_RecordingManager = openEPhys_DACQ.RecordingManager:main',
        'openEPhys_Processing = openEPhys_DACQ.Processing:main',
        'openEPhys_CreateAxonaData = openEPhys_DACQ.createAxonaData:main',
        'openEPhys_Configuration = openEPhys_DACQ.package_configuration:main'
    ]
}


def main():
    import io
    with io.open(os.path.join(HERE, 'README.md'), 'r') as readme:
        setup(
            name                 = __project__,
            version              = __version__,
            description          = __doc__,
            long_description     = readme.read(),
            long_description_content_type = 'text/markdown',
            classifiers          = __classifiers__,
            author               = __author__,
            author_email         = __author_email__,
            url                  = __url__,
            license              = [
                c.rsplit('::', 1)[1].strip()
                for c in __classifiers__
                if c.startswith('License ::')
            ][0],
            keywords             = __keywords__,
            packages             = find_packages(),
            include_package_data = True,
            install_requires     = __requires__,
            extras_require       = __extra_requires__,
            entry_points         = __entry_points__,
            python_requires      = '>=3',
        )


if __name__ == '__main__':
    main()
