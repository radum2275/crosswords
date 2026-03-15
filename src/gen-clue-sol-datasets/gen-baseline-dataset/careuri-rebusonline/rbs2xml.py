#/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import re

print >> sys.stderr, "rbs2xml version 1.1"

file = open(sys.argv[1], 'r')

title = file.readline()
nr_rows = int(file.readline())
nr_cols = int(file.readline())
grid = ""
for row in range(0, nr_rows):
    grid += file.readline().replace("|", "").replace("-","").replace("+","")
nr_info_lines = int(file.readline())
other_info = ""
for r in range(0, nr_info_lines):
    other_info += file.readline() 
horizontal_clues = ""
vertical_clues = ""
horizontal = False
vertical = False
dic = False
dictionary = ""
while True:
    line = file.readline()
    if 'ORIZONTAL' in line:
        horizontal = True
    if 'VERTICAL' in line:
        vertical = True
        horizontal = False
    if 'ionar: ' in line.lower():
        vertical = False
        dic = True
    if (vertical):
        vertical_clues += line
    if (horizontal):
        horizontal_clues += line
    if (dic):
        dictionary += line
    if not line: break

horizontal_clues = horizontal_clues.replace("ORIZONTAL:", "").strip();
vertical_clues = vertical_clues.replace("VERTICAL:" , "").strip();
horizontal_clues = re.sub(r"([^\.])(\.)( \d)", r"\1\3", horizontal_clues);
vertical_clues = re.sub(r"([^\.])(\.)( \d)", r"\1\3", vertical_clues);
horizontal_clues = re.sub(r"([^\.])(\.$)", r"\1", horizontal_clues);
vertical_clues = re.sub(r"([^\.])(\.$)", r"\1", vertical_clues);
#vertical_clues = re.sub(r"([a-z]|\))\.", r"\1", vertical_clues);
horizontal_clues = re.sub(r'(?<=[,])(?=[^\s])', r' ', horizontal_clues)
vertical_clues = re.sub(r'(?<=[,])(?=[^\s])', r' ', vertical_clues)
horizontal_clues = re.sub(r"(?<!\s)\(", r' (', horizontal_clues)
vertical_clues = re.sub(r"(?<!\s)\(", r' (', vertical_clues)
horizontal_clues = horizontal_clues.replace("...", "... ");
horizontal_clues = horizontal_clues.replace("…", "... ");
vertical_clues = vertical_clues.replace("...", "... ");
vertical_clues = vertical_clues.replace("…", "... ");
horizontal_clues = horizontal_clues.replace("    ", " ");
vertical_clues = vertical_clues.replace("    " , " ");
horizontal_clues = horizontal_clues.replace("   ", " ");
vertical_clues = vertical_clues.replace("   " , " ");
horizontal_clues = horizontal_clues.replace("  ", " ");
vertical_clues = vertical_clues.replace("  " , " ");
horizontal_clues = re.sub(r" – ", " - ", horizontal_clues);
vertical_clues = re.sub(r" – ", " - ", vertical_clues);
horizontal_clues = re.sub(r" !", "!", horizontal_clues);
vertical_clues = re.sub(r" !", "!", vertical_clues);
horizontal_clues = re.sub(r" \)", ")", horizontal_clues);
vertical_clues = re.sub(r" \)", ")", vertical_clues);
horizontal_clues = re.sub(r"\( ", "(", horizontal_clues);
vertical_clues = re.sub(r"\( ", "(", vertical_clues);
horizontal_clues = re.sub(r"[^1-9]1\)", "", horizontal_clues);
vertical_clues = re.sub(r"[^1-9]1\)" , "", vertical_clues);
horizontal_clues = re.sub(r" \d\d\)", " - ", horizontal_clues);
vertical_clues = re.sub(r" \d\d\)", " - ", vertical_clues);
horizontal_clues = re.sub(r" \d\)", " - ", horizontal_clues);
vertical_clues = re.sub(r" \d\)", " - ", vertical_clues);
horizontal_clues = re.sub(r"^\d\)", "", horizontal_clues);
vertical_clues = re.sub(r"^\d\)", "", vertical_clues);
print "<puzzle>"
print "<title>" + title.strip() + "</title>"
print "<author>Unknown</author>"
print "<rows>" + str(nr_rows) + "</rows>"
print "<columns>" + str(nr_cols) + "</columns>"
print "<grid>"
print grid.strip()
print "</grid>"
print "<horizontal_clues>"
print horizontal_clues.strip()
print "</horizontal_clues>"
print "<vertical_clues>"
print vertical_clues.strip()
print "</vertical_clues>"
if (len(dictionary) > 0):
    print "<dictionary>" + dictionary.split(":")[1].strip() + "</dictionary>"
print "<info_to_display>"
other_info = re.sub(r'(?<=[.,])(?=[^\s])', r' ', other_info.strip())
print other_info.strip()
print "</info_to_display>"
print "</puzzle>"

