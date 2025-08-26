from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='nitropulse',
        version='0.1.0',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        include_package_data=True,
        package_data={'nitropulse': ['config/gdd/*.json']},
        entry_points={
            'console_scripts': [
                'nitropulse = nitropulse.core:main',
            ],
        },
    )