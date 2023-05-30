from setuptools import setup, find_packages

setup(
  name = 'tree-of-thoughts',
  packages = find_packages(exclude=[]),
  version = '0.3.1',
  license='MIT',
  description = 'Tree of Thoughts - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/tree-of-thoughts',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers',
    "Prompt Engineering"
  ],
  install_requires=[
    'guidance',
    'openai',
    'transformers',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
