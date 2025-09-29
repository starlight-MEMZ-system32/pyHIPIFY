#!/usr/bin/env python
from setuptools import setup

if __name__ == '__main__':
    setup(
        name='pyHIPIFY',
        version='0.0.1b1',
        description='Simple utility to transform CUDA source into HIP source',
        long_description=open('README.md', encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        author='The Sunder islands',
        author_email='1317376584@qq.com',
        url='',
        license='MIT',
        package_data={
            'pyHIPIFY': ['*']
        },
        packages=[
            'pyHIPIFY',
        ],
        entry_points={
            'console_scripts': [
                'pyHIPIFY = pyHIPIFY.cli:main',
            ]
        },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Software Development :: Code Generators',
            'Topic :: Software Development :: Compilers',
        ],
        python_requires='>=3.6',
        setup_requires=['setuptools'],
        install_requires=[],
        keywords='cuda hip rocm amd gpu',
    )