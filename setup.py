import setuptools

setuptools.setup(
                 name='qfast',
                 version='0.0.1',
                 description='Quantum Fast Approximate Synthesis Tool',
                 url='https://github.com/edyounis/qfast',
                 author='Ed Younis',
                 author_email='edyounis@berkeley.edu',
                 packages=['qfast'],
                 zip_safe=False,
                 install_requires=[ 'qiskit' ]
                )
