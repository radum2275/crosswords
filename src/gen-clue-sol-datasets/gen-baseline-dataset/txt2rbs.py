#/usr/bin/python
# -*- coding: utf -*-

import sys
import re

have_read_title = False
grid = ""
other_info = ""
horiz_clues = ""
vert_clues = ""
read_horiz = False
read_vert = False
read_dic = False
dictionary = ""
read_grid = False

sys.stderr.write("Processing file %s\n" %sys.argv[1])

lines = [line.strip() for line in open(sys.argv[1], 'r')]
for line in lines:
    if line == "":
        read_horiz = False
        read_vert = False
        read_dic = False
        read_grid = False
        continue
    if ("----" in line):
        read_horiz = False
        read_vert = False
        read_dic = False
        continue
    if "|" in line:
        read_grid = True
    if "|" not in line and not have_read_title:
        title = line
        have_read_title = True
        continue
    if 'ORIZONTAL' in line:
        read_horiz = True
        read_vert = False
        read_dic = False
    if 'VERTICAL' in line:
        read_vert = True
        read_horiz = False
        read_dic = False
    if 'ionar: ' in line.lower():
        read_vert = False
        read_horiz = False
        read_dic = True
    if (read_vert):
        vert_clues += " " + line
    if (read_horiz):
        horiz_clues += " " + line
    if (read_dic):
        dictionary += line
    if ("|" in line):
        grid += line
    read_other = have_read_title and not read_grid and not read_vert and not read_horiz
    if read_other:
        other_info += line + "\n"
    if not line: break

grid = grid.replace("| | | |","|.|.|.|")
grid = grid.replace("| | |","|.|.|")
grid = grid.replace("| |","|.|")
grid = grid.replace("||","|.|")
grid = grid.replace("|","")
grid = grid.replace(" ","")
newgrid = ""
for idx in range(0,len(grid)):
    if not grid[idx].isdigit():
        if idx < len(grid)-1 and grid[idx] == '.' and grid[idx+1].isdigit():
            newgrid = newgrid + ('\n')
        else:
            if (idx < len(grid) - 1 or grid[idx] != '.'):
                newgrid = newgrid + (grid[idx])
if len(newgrid) > 2 and newgrid[0] == '.' and newgrid[1] == '\n':
    newgrid = newgrid[2:]

nr_rows = max(map(int, re.findall(r'\d+', horiz_clues)))
nr_cols = max(map(int, re.findall(r'\d+', vert_clues)))
horiz_clues = horiz_clues.replace(":",": ")
vert_clues = vert_clues.replace(":",": ")
horiz_clues = horiz_clues.replace(".",". ")
vert_clues = vert_clues.replace(".",". ")
horiz_clues = horiz_clues.replace(" )",")")
vert_clues = vert_clues.replace(" )",")")
horiz_clues = horiz_clues.replace(" .",".")
vert_clues = vert_clues.replace(" .",".")
horiz_clues = horiz_clues.replace(" ,",",")
vert_clues = vert_clues.replace(" ,",",")
horiz_clues = horiz_clues.replace(",",", ")
vert_clues = vert_clues.replace(",",", ")
horiz_clues = horiz_clues.replace("("," (")
vert_clues = vert_clues.replace("("," (")
horiz_clues = horiz_clues.replace("–"," - ")
vert_clues = vert_clues.replace("–"," - ")
horiz_clues = horiz_clues.replace("…","... ")
vert_clues = vert_clues.replace("…","... ")
horiz_clues = horiz_clues.replace(" ...","...")
vert_clues = vert_clues.replace(" ...","...")
horiz_clues = ' '.join(horiz_clues.split())
vert_clues = ' '.join(vert_clues.split())
print (title.strip())
print (str(nr_rows))
print (str(nr_cols))
print (newgrid.strip())

other_info = other_info.replace("“","")
other_info = other_info.replace("”","")
other_info = other_info.replace("„","")
other_info = other_info.replace(",",", ")
other_info = other_info.replace("&",", ")
other_info = other_info.replace(" ,",",")
other_info = other_info.replace("  "," ")
other_info = other_info.replace("  "," ")
print (other_info.count("\n"))
print (other_info.strip())
print ("")
print (horiz_clues)
print (vert_clues)
if (len(dictionary) > 0):
    print (dictionary)

