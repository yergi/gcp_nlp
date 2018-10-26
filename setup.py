from distutils.core import setup

with open('readme.md', 'r') as f:
    readme = f.read()

setup(
    name='gcp_nlp',
    version='0.1.0',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='James Colvin',
    packages=['gcp_nlp'],
    install_requires=['gcsfs==0.1.2', 'google-cloud-automl==0.1.1']
)
