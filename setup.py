import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="AIStructDynSolve",  # 模块名称
    version="1.14",  # 当前版本
    author="Ke Du(杜轲)",  # 作者
    author_email="duke@iem.ac.cn",  # 作者邮箱
    description="AI-powered framework to solve both forward and inverse problems in structural dynamics",  # 模块简介
    long_description=long_description,  # 模块详细介绍
    long_description_content_type="text/markdown",  # 模块详细介绍格式
    url="https://github.com/duke-iem/AIStructDynSolve.git",  # 模块github地址
    packages=setuptools.find_packages(),  # 自动找到项目中导入的模块
    # 模块相关的元数据
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit-optimize',
        'scipy',
        'pandas',
        'openpyxl',
    ],
    python_requires='>=3.10',
)


