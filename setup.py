from setuptools import setup, find_packages

setup(
    name="cs682-course-project",
    version="0.1",
    packages=find_packages(exclude=["datasets", "docs", "embeddings", "retrieved_items", "static", "metrics"]),
)