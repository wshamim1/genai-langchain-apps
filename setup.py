"""
Setup configuration for genai-langchain package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="genai-langchain",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive framework for working with multiple chat model providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genai-langchain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="langchain, chat, ai, llm, openai, anthropic, google, gemini, gpt, claude",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/genai-langchain/issues",
        "Source": "https://github.com/yourusername/genai-langchain",
        "Documentation": "https://github.com/yourusername/genai-langchain#readme",
    },
)

# Made with Bob
