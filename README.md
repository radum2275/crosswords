# crosswords
LLMs and crossword puzzles

## Important Datasets

We have generated two datasets with clues and solutions, both with a similar structure, and both available in folder `data`:

- The main dataset is `extracted_data.json`, which is the benchmark that we propose for LLM evaluation. It contains clues and solutions from published crosswords puzzles where clues are meant to trick the reader. Often they have two more more meanings, where the more obvious meaning is wrong (i.e., it doesn't match the solution), and a second, correct meaning is more subtle to find.
- The baseline dataset is `baseline-dataset.json`. It contains general knowledge questions and answers extracted from thematic puzzles.

## Project Structure

- Folder `gen-clue-sol-datasets` has code used to generate the main dataset (subfolder `gen-main-dataset`) and the baseline dataset (subfolder `gen-baseline-dataset`). Subfolder `gen-baseline-dataset` has a detailed README file on its own. No detailed knowledge of folder `gen-clue-sol-dataset` is required, since their intended results, namely files `extracted_data.json` and `baseline-dataset.json`, are already generated and made available in folder `data`.

## Manual Annotation of LLM Explanations

Part of the LLM explanations have been manually checked with a process that contains the following steps:

- Convert `*.json` files in folder `data` to `csv`. This is achieved with command

```bash
for i in ../../data/*.json ; do python json2csv.py $i ; done
```

  An input file `<NAME>.json` leads to the output file`<NAME>_output.csv`. E.g., `out_gpt-oss_v1_0.json` to `out_gpt-oss_v1_0_output.csv`.

- Out of all `*.csv` files, we selected the three files with the richest hints to the LLM (i.e., those that give the first four letters of the solutions). We selected these files under the assumption that they would have a higher percentage of correct predictions, compared to the versions with less detailed hints. 

- We created separate copies of these three files, to avoid their accidental overwriting with a new run of the command shown at step 1. The three resulting files are called
``out_gpt-oss_v3_4_output_annotated.csv``, ``out_llama_v3_4_output_annotated.csv`` and ``out_granite_v3_4_output_annotated.csv``.

- In each of these three files, we filtered records with correct predictions and manually annotated these records in a column called `rationale_score`, with values from 0 to 5. A value of 0 or 1 means no or poor relevance of the explanation. A value of 2 or 3 means a partial relevance. Values of 4 and 5 signify a good explanation. A few records with correct predictions received no rationale score. For example, the no score was given when the answer in the input was wrong (typo), which was observed in at least one case.

Summary statistics are printed with script `count_csv_stats.py`:

```bash
python count_csv_stats.py ../../data/out_gpt-oss_v3_4_output_annotated.csv 
Total records: 200
Records with match = True: 163
Records with rationale_score >= 5: 81

python count_csv_stats.py ../../data/out_granite_v3_4_output_annotated.csv 
Total records: 199
Records with match = True: 95
Records with rationale_score >= 5: 23

python count_csv_stats.py ../../data/out_llama_v3_4_output_annotated.csv 
Total records: 200
Records with match = True: 119
Records with rationale_score >= 5: 50
```
