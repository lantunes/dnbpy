from setuptools import setup

setup(name="dnbpy",
      version="0.3",
      description="DnBPy, A Dots 'n Boxes Game Engine for Python",
      long_description="DnBPy is a light-weight Dots 'n Boxes game engine for Python."
                       "It is particularly useful for AI projects, and can be used "
                       "as an environment for Reinforcement Learning projects.",
      license="MIT",
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/dnbpy',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=["dnbpy"],
      keywords=["dots and boxes", "game environment", "reinforcement learning"],
      python_requires='>3.5.2')
