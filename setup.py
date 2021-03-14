from setuptools import setup

descr = """Cellpose-Planer is the cellpose models on planer framework"""

if __name__ == '__main__':
    setup(name='cellpose-planer',
        version='0.14',
        url='https://github.com/Image-Py/cellpose-planer',
        description='Cellpose-Planer is the cellpose models on planer framework',
        long_description=descr,
        author='Y.Dong, YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=['cellpose_planer'],
        package_data={},
        install_requires=[
            'numpy',
            'scipy',
            'planer'
        ],
    )
