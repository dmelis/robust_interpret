from setuptools import setup, find_packages

setup(name='robust_interpret',
      version='0.1',
      description='Perturbation and gradient-based methods for Deep Network interpretability',
      url='https://github.mit.edu/dalvmel/robust_interpret',
      author='David Alvarez Melis (MIT)',
      author_email='dalvmel@mit.edu',
      license='MIT',
      packages=find_packages(),#('src'),
      #package_dir = {'': 'src'},
      #package_dir = {'robust_interpret': 'src'},
      include_package_data=True,
      install_requires=[
            'scipy',
            'numpy',
            'matplotlib',
            'scikit-image',
            'scikit-optimize'
      ],
      extras_require={
            "tf": ["tensorflow>=1.0.0"],
            "tf_gpu": ["tensorflow-gpu>=1.0.0"],
            # "lime",
            # "shap"
      },
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
