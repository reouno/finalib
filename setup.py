#!/usr/bin/env python

from distutils.core import setup

setup(name='finalib',
      version='0.2.2',
      install_requires=[
            'numpy',
            'pandas',
            'debtcollector',      
      ],
      description='Python finance utilities',
      author='UNO Leo',
      author_email='leouno12@gmail.com',
      url='https://github.com/tmtlu/finalib',
      packages=['finalib', 'finalib.mine'],
     )