from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.readlines()

setup(
    name='cog_ieeg',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Alex Lepauvre',
    author_email='alex.lepauvre@ae.mpg.de',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Cogitate-consortium/iEEG-data-release.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
