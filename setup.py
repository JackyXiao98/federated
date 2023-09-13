from setuptools import setup

setup(
    name='federated',
    version='1.0',
    description='federated',
    url='https://github.com/JackyXiao98/federated',
    author='Yingtai Xiao',
    packages=['multi_epoch_dp_matrix_factorization'],
    zip_safe=False,
    install_requires=['tensorflow_privacy==0.61.0',
                      ],
)