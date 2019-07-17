# Peeps Finder
Daniele Moro

A tool for automatically extracting contact attributes from the internet. 
This is the final project for CS 536 Natural Language Processing.

[View the final paper](https://github.com/danielemoro/PeepsFinder/blob/master/PeepsFinderPaper.pdf)

You can find videos demonstrating this tool [here](https://youtu.be/eyu2XChnFaQ) and [here](https://youtu.be/5gnSc_UnwoQ)
A slide deck describing Peeps Finder can be found [here](https://docs.google.com/presentation/d/1EKwgafM0n_7IBOFMByxOKBiCF50Gv93bEc4kCT9tHOQ/edit?usp=sharing)

# Requirements
This project has only been tested with the following:
- Windows 10 
- GeForce GTX 1080 Graphics Card

# Installation
1. Install Python 3.6.8 using [Anaconda](https://www.anaconda.com/distribution/)
2. Install all the python packages with the version found in `requirements.txt`
3. Download and set up the code to run the relationship extraction model. 
For this project, I am leveraging `Context-Aware Representations for Knowledge Base Relation Extraction`.
You can find the GitHub repository [here](https://github.com/UKPLab/emnlp2017-relation-extraction)
4. Edit the first line of `peeps_finder.py` with the path to the root directory of the relationship extraction model.

# Usage
- For basic usage of Peeps Finder, run `python main.py` on the command line. 
Then please respond to the following prompts. 
When completed, the program will write the verified contact information to a json file.
- To use Peeps Finder with a web-based user interface, please download and set up the corresponding web server. 
You can find the GitHub repository [here](https://github.com/danielemoro/peeps/tree/peeps_finder). 
After completing this, edit the first two lines of `server.py` 
and provide the file paths for communication with the web server. 
Start the web server, then start Peeps Finder by entering `python server.py` on your command line. 
