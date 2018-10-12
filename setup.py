try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

setup(name='galtag',
      version='0.1',
      description='Galaxy tagging, a photometric redshift refinement code',
      url='',
      author='pkaf',
      author_email='pkafauthor@gmail.com',
      license='MIT',
      packages=['examples', 'docs', 'galtag'],
      zip_safe=False)

