# Best Match 25 (BM25) Keyword Search of Movie Database
## 1) Introduction
This code employs a BM25 keyword search on a database of 5000 movies (or any database with identical structure). A number of simpler search related commands are also supplied:
- Basic term match with results ordered by document id
- Term frequency (TF) ordered from highest to lowest
- Inverse document frequency (IDF) ordered from highest to lowest
- Combined TF-IDF (multiplication of the two) ordered from highest to lowest
- BM25 specific TF ordered from highest to lowest
- BM25 specific IDF ordered from highest to lowest
- Full BM25 score ordered from highest to lowest

Top 5 matches are supplied by default.

The BM25 search algorithm, when coupled with text preprocessing, allows for effective retrieval of relevant items. As applied to the movie database, it searches both the title and the description of the movie. Using BM25, the advantage of a longer description (more occurrences of key words) is less beneficial. BM25 also gives more weight to words that occur less frequently in the database.

The following equations are used in the implementation of the BM25 search algorithm:<br>
<p align="center"><img src="./images/Lnorm.svg"></p>
<p align="center"><img src="./images/bm25_tf.svg"></p>
<p align="center"><img src="./images/bm25_idf.svg"></p>
<p align="center"><img src="./images/bm25_score.svg"></p>

**Where:**<br>
N = total number of movies<br>
DF = document frequency (nb of documents containing the term)<br>
TF = term frequency (number of occurrences of term within a given document)<br>
L<sub>norm</sub> = normalized document length<br>
k<sub>1</sub> = saturation constant<br>
b = normalization constant (between 0 and 1)<br>

## 2) Requirements
### Virtual environment
- Python package manager: uv 0.9.11
- Python version: ≥3.13

### Installation
- Install uv (if necessary):
    - **curl -LsSf https://astral.sh/uv/0.9.11/install.sh | sh**
- Perform a git pull of the project
- Navigate to the local git project folder in a terminal
- Add uv to the path:
    - **source $HOME/.local/bin/env**
- Build the environment:
    - **uv build**

### Movie data:
Movie data must be downloaded separately and placed in the /data repository of the project. It can be downloaded at the following link:<br>
https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json

### Hardware and operating system:
This project should operate on any system capable of using the above uv and python versions. It was tested on macOS 13.7.8 with Intel Core i7 processor.

### Adapting to another database
- Use a json file having the same structure as the movie database.
- Adjust  **cli/lib/search_utils.py** as needed:
    - Modify the **load_database** function
    - Adjust the file name in the **DATA_PATH** variable

## 3) Using the files
- Navigate to the local git project folder in a terminal
- Build the cached database files:
    - **uv run cli/keyword_search_cli.py build**
- Type a command, for example:
    - **uv run cli/keyword_search_cli.py bm25search "space adventure"**
- Go to cli/lib/search_utils.py to adjust the default search limit, k1 and b values.
- Call help function to get details of possible commands:
    - **uv run ./cli/keyword_search_cli.py --help**
    - **uv run cli/keyword_search_cli.py bm25search --help**

## 4) References
This project was developped while following an online course from boot.dev for retrieval augmented generation:<br>
https://www.boot.dev/lessons/7a92d1c1-d202-481a-ae5f-14fc9f97b640