import setuptools

setuptools.setup(
    name="carformer",
    version="0.2",
    author="Shadi Hamdan et al.",
    author_email="shamdan17@ku.edu.tr",
    description="Transformers for Sequential Decision making in Autonomous Driving",
    long_description="Transformers for Sequential Decision making in Autonomous Driving",
    long_description_content_type="text",
    url="https://github.com/Shamdan17/ETA",
    project_urls={
        "Bug Tracker": "https://github.com/Shamdan17/ETA/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
)
