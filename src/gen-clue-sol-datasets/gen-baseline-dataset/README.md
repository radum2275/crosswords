
# Baseline Dataset

The main purpose of work captured in the folder at hand is creating the file `baseline-dataset.json`. This has the same structure as file `extracted-data.json`, which is the main dataset (i.e., the dataset that we propose as a novel benchmark). The identical structuring allows for an easier and fair comparison, running the same experiments with both datasets.

The initial raw data we start from are thematic puzzles provided by the Rebus Ideal editor-in-chief, present in folders `doc` and `docx`. Specifically, we used part of the puzzles present in folders `doc`.

The code present in this folder was copy-pasted from rebusonline and adapted to the needs at hand.

## Steps performed to generate the baseline dataset

- Parsed input Word doc files to obtain txt files and possibly incorrect rbs files with command `create-raw-rbs.bash`. The main result is a set of possibly incorrect `*.rbs` files in folder `processed-rbs`.

- Performed a first manual correction of automatically generated `*.rbs` files available in folder `processed-rbs`.

- Copied some `*.rbs` files from `processed-rbs` (namely, `*.rbs` that required a smaller amount of fixing) to folder `tematice-ideal/careuri-rebusonline`. 

All the remaining steps, presented below, are performed inside folder `tematice-ideal/careuri-rebusonline`.

- Processed `*.rbs` files to extract (clue, solution) pairs, using command:
```bash
  for i in *.rbs ; do ./allsteps.bash $i ; done > raw-data-pairs.txt 2>&1
```

- Copied `raw-data-pairs.txt` into `filtered-data-pairs.txt`, and manually filtered records inside ``filtered-data-pairs.txt`` with rules such as the following:
  - Remove records where the solution is an abbreviation
  - Remove records with the exclamaiton mark at the end of the clue, which indicate a play on letters
  - Remove records where the clue is too vague
  - Remove records where the subject (clue and solution) are considered to be very obscure 
  - Remove some records where the solution has two words

Kept fixing `*.rbs` files in folder `tematice-ideal/careuri-rebusonline` as errors were still detected. Repeated the last two steps as needed until no errors were detected.

- Finally, copy the resulting file  `baseline-dataset.json` to the `data` folder of the main project.