from setuptools import setup, find_packages


# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        # Exclude any comments or empty lines
        return [line.strip() for line in req if line.strip() and not line.startswith('#')]


# Call the function and store the requirements list
install_requires = read_requirements()

setup(
    name='gemini_toolbox',
    version='0.3.0',
    packages=find_packages(),
    description='Toolbox For Using Gemini Agents SDK',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    author='Viacheslav Kovalevskyi',
    author_email='viacheslav@kovalevskyi.com',
    entry_points={
        'console_scripts': [
            'gt = gemini_toolbox.bin.pipe:main',  
        ],
    },
    license='MIT',
    install_requires=install_requires
)