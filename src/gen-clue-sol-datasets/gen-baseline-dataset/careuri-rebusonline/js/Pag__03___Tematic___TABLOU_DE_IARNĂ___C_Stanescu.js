// File structure based on [c] Jesse Weisbeck, MIT/GPL
(function($) {
  $(function() {
    var puzzleData = 
[     {'answer': 'ZAPADA', 'clue': '1)\t\tPrecipitație atmosferică solidă, alcătuită din cristale de apă înghețată, care cade pe sol la temperaturi sub zero grade Celsius, și formează un strat alb, granular', 'orientation': 'across', 'position': 1, 'startx': 1, 'starty': 1},
      {'answer': 'RAM', 'clue': '--\t\tCreangă a copacului care a rămas fără frunze pe timpul iernii', 'orientation': 'across', 'position': 2, 'startx': 8, 'starty': 1},
      {'answer': 'ALUNECOASA', 'clue': '2)\t\t Suprafață formată din gheață, fără aderență, pe care se alunecă foarte ușor', 'orientation': 'across', 'position': 3, 'startx': 1, 'starty': 2},
      {'answer': 'GIC', 'clue': '3)\t\t „Global Instructional Chemistry” (acronim)', 'orientation': 'across', 'position': 4, 'startx': 1, 'starty': 3},
      {'answer': 'MENIUC', 'clue': '--\t\t„Stepa mea! În depărtarea de zăpadă/ Vântul își doinește viața lui nomadă”, este un citat din poezia „Iarna” oferită tiparului de acest eseist, poet, prozator, publicist și traducător din Republica Moldova', 'orientation': 'across', 'position': 5, 'startx': 5, 'starty': 3},
      {'answer': 'AF', 'clue': '4)\t\t Afară devreme!', 'orientation': 'across', 'position': 6, 'startx': 1, 'starty': 4},
      {'answer': 'VE', 'clue': '--\t\tCap de veveriță!', 'orientation': 'across', 'position': 7, 'startx': 4, 'starty': 4},
      {'answer': 'OCDE', 'clue': '--\t\t„Organización de Cooperación y Desarrollo Económico” (siglă)', 'orientation': 'across', 'position': 8, 'startx': 7, 'starty': 4},
      {'answer': 'RACITA', 'clue': '5)\t\t Perioadă caracterizată de temperaturi scăzute, cu risc crescut de afecțiuni respiratorii', 'orientation': 'across', 'position': 9, 'startx': 1, 'starty': 5},
      {'answer': 'UAD', 'clue': '--\t\t„Under Agreement Dated” (acronim)', 'orientation': 'across', 'position': 10, 'startx': 8, 'starty': 5},
      {'answer': 'ANUAR', 'clue': '6)\t\t Publicație tipărită de regulă la sfârșitul unui an calendaristic, care poate fi o prezentare detaliată a activității unei instituții sau organizații, sau chiar a datelor statistice importante dintr- o ramură sau țară de pe parcursul anului', 'orientation': 'across', 'position': 11, 'startx': 1, 'starty': 6},
      {'answer': 'TO', 'clue': '--\t\tTopită de la început!', 'orientation': 'across', 'position': 12, 'startx': 9, 'starty': 6},
      {'answer': 'NEIOS', 'clue': '7)\t\t Acoperit cu multă zăpadă', 'orientation': 'across', 'position': 13, 'startx': 4, 'starty': 7},
      {'answer': 'MINUS', 'clue': '8)\t\t Ambient cu temperatură sub zero grade', 'orientation': 'across', 'position': 14, 'startx': 1, 'starty': 8},
      {'answer': 'OROS', 'clue': '--\t\tPictoriță autohtonă prezentă pe simezele expozițiilor cu tabloul acrilic pe pânză intitulat „Timp înghețat” (Mariana)', 'orientation': 'across', 'position': 15, 'startx': 7, 'starty': 8},
      {'answer': 'OSO', 'clue': '9)\t\t „Orbiting Solar Observatory” (acronim)', 'orientation': 'across', 'position': 16, 'startx': 1, 'starty': 9},
      {'answer': 'CER', 'clue': '--\t\tSpațiul vast aflat deasupra pământului, care iarna, este în majoritatea timpului acoperit de nori', 'orientation': 'across', 'position': 17, 'startx': 5, 'starty': 9},
      {'answer': 'RK', 'clue': '--\t\tRiley King', 'orientation': 'across', 'position': 18, 'startx': 9, 'starty': 9},
      {'answer': 'RAU', 'clue': '10)\t\t Volumul „Omul de zăpadă” face parte din opera acestui poet și prozator transilvănean (Aurel)', 'orientation': 'across', 'position': 19, 'startx': 3, 'starty': 10},
      {'answer': 'ELEI', 'clue': '--\t\tExclamație prin care se interpelează ascultătorul (pop.)', 'orientation': 'across', 'position': 20, 'startx': 7, 'starty': 10},
      {'answer': 'ZAGARA', 'clue': '1)\t\tBlană de animal prelucrată, folosită adesea pentru a face marginea unei căciuli sau guler la diferite articole de îmbrăcăminte pentru iarnă', 'orientation': 'down', 'position': 1, 'startx': 1, 'starty': 1},
      {'answer': 'MOS', 'clue': '--\t\tPersonaj legendar din folclor care aduce cadouri copiilor cuminți cu ocazia sărbătorilor de iarnă', 'orientation': 'down', 'position': 14, 'startx': 1, 'starty': 8},
      {'answer': 'ALIFANTIS', 'clue': '2)\t\t Muzician, actor, poet, cântăreț și compozitor român, din repertoriul căruia amintim melodia „Decembre” (Nicu)', 'orientation': 'down', 'position': 21, 'startx': 2, 'starty': 1},
      {'answer': 'PUC', 'clue': '3)\t\t Obiect din cauciuc dur întrebuințat în loc de minge la jocul de hochei pe gheață', 'orientation': 'down', 'position': 22, 'startx': 3, 'starty': 1},
      {'answer': 'CU', 'clue': '--\t\tPicură pe mijloc!', 'orientation': 'down', 'position': 23, 'startx': 3, 'starty': 5},
      {'answer': 'NOR', 'clue': '--\t\tMasă de vapori suspendată și delimitată în atmosferă, sub formă de picături fine de apă sau cristale de gheață', 'orientation': 'down', 'position': 24, 'startx': 3, 'starty': 8},
      {'answer': 'AN', 'clue': '4)\t\t Se schimbă în noaptea de revelion (sg.)', 'orientation': 'down', 'position': 25, 'startx': 4, 'starty': 1},
      {'answer': 'VIANU', 'clue': '--\t\tCritic, istoric literar, poet și traducător giurgiuvean, autorul poeziei „Norii”, din care citam: „Și de cădeți cu rodul unei ploi/ Sau cu-a zăpezii albă risipire, / Sunteți avântul către forme noi” (Tudor)', 'orientation': 'down', 'position': 7, 'startx': 4, 'starty': 4},
      {'answer': 'DEMETRESCU', 'clue': '5)\t\t Poemul „Tablou de iarnă” este creația lirică a acestui poet presimbolist craiovean (Traian)', 'orientation': 'down', 'position': 26, 'startx': 5, 'starty': 1},
      {'answer': 'ACE', 'clue': '6)\t\t Cristale de gheață lungi și ascuțite', 'orientation': 'down', 'position': 27, 'startx': 6, 'starty': 1},
      {'answer': 'ONO', 'clue': '7)\t\t Artistă și muziciană japonezo-americană, interpreta melodiei „Ascultă, ninge!” (Yoko)', 'orientation': 'down', 'position': 28, 'startx': 7, 'starty': 2},
      {'answer': 'MOORE', 'clue': '--\t\tRenumit actor englez, prezent în distribuția filmului „Crăciun la castel” regizat de Michael Damian (Roger)', 'orientation': 'down', 'position': 29, 'startx': 7, 'starty': 6},
      {'answer': 'RAICU', 'clue': '8)\t\t Poet, prozator, traducător și jurnalist român, autorul volumului de proză scurtă intitulat „Hai cu mine!”, din care amintim titlul „Cărări de munte înzăpezite” (Alexandru)', 'orientation': 'down', 'position': 2, 'startx': 8, 'starty': 1},
      {'answer': 'SR', 'clue': '--\t\tSere!', 'orientation': 'down', 'position': 30, 'startx': 8, 'starty': 7},
      {'answer': 'ASUDAT', 'clue': '9)\t\t Aburit', 'orientation': 'down', 'position': 31, 'startx': 9, 'starty': 1},
      {'answer': 'ORE', 'clue': '--\t\tUnități de măsură ale timpului, care sunt peste 15 la număr, în cea mai lungă noapte a anului, cea a solstițiului', 'orientation': 'down', 'position': 32, 'startx': 9, 'starty': 8},
      {'answer': 'MACEDONSKI', 'clue': '10)\t\t „Pustie și albă e-ntinsa câmpie... / Sub viscolu- albastru ea geme cumplit... ”, sunt versuri preluate din „Noaptea de decembrie”, creație lirică a acestui poet, prozator, dramaturg și publicist român. (Alexandru)', 'orientation': 'down', 'position': 33, 'startx': 10, 'starty': 1}]
    $('#puzzle-wrapper').crossword(puzzleData, `TABLOU DE IARNĂ`, 'Constantin BRAȘOVEANU – Ploiești');
  })
})(jQuery)
